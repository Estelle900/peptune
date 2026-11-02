import os
import sys
import time
from pathlib import Path

import hydra
import omegaconf
import pandas as pd
import torch

from diffusion import Diffusion
from generate_mcts import PROTEIN_SEQUENCES
from LNR_Sequences import LNR_Sequences as LNR_SEQUENCES
from scoring.scoring_functions import ScoringFunctions
from tokenizer.my_tokenizers import SMILES_SPE_Tokenizer
from utils.app import PeptideAnalyzer
from utils.generate_utils import mask_for_de_novo

# Hydra resolvers used by the shared configuration.
omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd, replace=True)
omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count, replace=True)
omegaconf.OmegaConf.register_new_resolver('eval', eval, replace=True)
omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y, replace=True)


@torch.no_grad()
def generate_sequence_unconditional(config, sequence_length: int, mdlm: Diffusion):
    inputs = mdlm.tokenizer.encode(mask_for_de_novo(config, sequence_length))
    inputs = {key: value.to(mdlm.device) for key, value in inputs.items()}
    generated = mdlm._sample(x_input=inputs)
    return generated, inputs


def protein_sequence(name):
    if name is None:
        return None
    key = str(name).lower()
    sequence = PROTEIN_SEQUENCES.get(key, LNR_SEQUENCES.get(key))
    if sequence is None:
        supported = sorted(set(PROTEIN_SEQUENCES) | set(LNR_SEQUENCES))
        raise ValueError(f"Unknown target protein {name!r}. Supported targets: {', '.join(supported)}")
    return sequence


@hydra.main(version_base=None, config_path='.', config_name='config')
def main(config):
    base_path = Path(config.base_path).expanduser().resolve()
    target_name = str(config.get('prot_name1', 'gfap')).lower()
    second_name = config.get('prot_name2', None)
    second_name = str(second_name).lower() if second_name else None
    mode = config.get('mode', '2')
    model_type = config.get('model_type', 'unconditional')
    length = config.get('length', '100')
    epoch = config.get('epoch', '7')
    filename = f'{mode}_{model_type}_length_{length}_epoch_{epoch}'
    if second_name:
        filename += f'_vs_{second_name}'

    prot_seqs = [protein_sequence(target_name)]
    if second_name:
        prot_seqs.append(protein_sequence(second_name))

    tokenizer = SMILES_SPE_Tokenizer(
        str(base_path / 'src/tokenizer/new_vocab.txt'),
        str(base_path / 'src/tokenizer/new_splits.txt'),
    )
    checkpoint = str(config.eval.checkpoint_path)
    mdlm = Diffusion.load_from_checkpoint(
        checkpoint, config=config, tokenizer=tokenizer,
        strict=False, map_location='cpu', weights_only=False,
    )
    mdlm.eval()
    mdlm.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Use the same objective set as the existing unconditional generator.
    score_func_names = [
        'binding_affinity1', 'solubility', 'hemolysis',
        'nonfouling', 'permeability',
    ]
    if second_name:
        score_func_names.append('binding_affinity2')
    score_functions = ScoringFunctions(score_func_names, prot_seqs, device=mdlm.device)
    analyzer = PeptideAnalyzer()

    columns = [
        'Generated SMILES', 'Perplexity', 'Peptide Sequence',
        'Binding Affinity', 'Solubility', 'Hemolysis',
        'Nonfouling', 'Permeability',
    ]
    if second_name:
        columns.append('Binding Affinity 2')

    num_sequences = int(config.sampling.num_sequences)
    if num_sequences <= 0:
        raise ValueError('sampling.num_sequences must be positive')
    results = []
    num_valid = 0
    start = time.time()
    print(f'Target: {target_name}; second target: {second_name}; samples: {num_sequences}')
    print(f'Checkpoint: {checkpoint}')
    for i in range(num_sequences):
        generated, masked_inputs = generate_sequence_unconditional(
            config, int(config.sampling.seq_length), mdlm,
        )
        generated = generated.to(mdlm.device)
        perplexity = round(float(mdlm.compute_masked_perplexity(
            generated, masked_inputs['input_ids'])), 4)
        smiles = tokenizer.decode(generated)
        if analyzer.is_peptide(smiles):
            aa_sequence, seq_length = analyzer.analyze_structure(smiles)
            scores = score_functions(input_seqs=[smiles])[0]
            results.append([smiles, perplexity, aa_sequence, *map(float, scores)])
            num_valid += 1
            print(f'{i + 1}/{num_sequences}: valid | length={seq_length} | '
                  f'perplexity={perplexity} | scores={scores} | SMILES={smiles}')
        else:
            print(f'{i + 1}/{num_sequences}: invalid | '
                  f'perplexity={perplexity} | SMILES={smiles}')
        sys.stdout.flush()

    output_dir = base_path / 'results_unconditional' / target_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{filename}.csv'
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(output_path, index=False)
    print(f'Valid peptide fraction: {num_valid / num_sequences:.4f} '
          f'({num_valid}/{num_sequences})')
    print(f'Generation complete in {time.time() - start:.1f}s; saved {output_path}')
    if not df.empty:
        for column in ['Perplexity', *columns[3:]]:
            values = pd.to_numeric(df[column], errors='coerce').dropna()
            if not values.empty:
                print(f'{column}: mean={values.mean():.4f}, std={values.std():.4f}, '
                      f'min={values.min():.4f}, max={values.max():.4f}')


if __name__ == '__main__':
    main()


