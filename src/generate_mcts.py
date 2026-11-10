#!/usr/bin/env
import time
import torch
import torch.nn.functional as F
import math
import random
import sys
import pandas as pd
from utils.generate_utils import mask_for_de_novo
from diffusion import Diffusion
from pareto_mcts import Node, MCTS
import hydra
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, pipeline
from tokenizer.my_tokenizers import SMILES_SPE_Tokenizer
from utils.app import PeptideAnalyzer
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import numpy as np
# LNR testset
from LNR_Sequences import LNR_Sequences as LNR_SEQUENCES

# Protein sequence dictionary
PROTEIN_SEQUENCES = {
    'amhr': 'MLGSLGLWALLPTAVEAPPNRRTCVFFEAPGVRGSTKTLGELLDTGTELPRAIRCLYSRCCFGIWNLTQDRAQVEMQGCRDSDEPGCESLHCDPSPRAHPSPGSTLFTCSCGTDFCNANYSHLPPPGSPGTPGSQGPQAAPGESIWMALVLLGLFLLLLLLLGSIILALLQRKNYRVRGEPVPEPRPDSGRDWSVELQELPELCFSQVIREGGHAVVWAGQLQGKLVAIKAFPPRSVAQFQAERALYELPGLQHDHIVRFITASRGGPGRLLSGPLLVLELHPKGSLCHYLTQYTSDWGSSLRMALSLAQGLAFLHEERWQNGQYKPGIAHRDLSSQNVLIREDGSCAIGDLGLALVLPGLTQPPAWTPTQPQGPAAIMEAGTQRYMAPELLDKTLDLQDWGMALRRADIYSLALLLWEILSRCPDLRPDSSPPPFQLAYEAELGNTPTSDELWALAVQERRRPYIPSTWRCFATDPDGLRELLEDCWDADPEARLTAECVQQRLAALAHPQESHPFPESCPRGCPPLCPEDCTSIPAPTILPCRPQRSACHFSVQQGPCSRNPQPACTLSPV',
    'tfr': 'MMDQARSAFSNLFGGEPLSYTRFSLARQVDGDNSHVEMKLAVDEEENADNNTKANVTKPKRCSGSICYGTIAVIVFFLIGFMIGYLGYCKGVEPKTECERLAGTESPVREEPGEDFPAARRLYWDDLKRKLSEKLDSTDFTGTIKLLNENSYVPREAGSQKDENLALYVENQFREFKLSKVWRDQHFVKIQVKDSAQNSVIIVDKNGRLVYLVENPGGYVAYSKAATVTGKLVHANFGTKKDFEDLYTPVNGSIVIVRAGKITFAEKVANAESLNAIGVLIYMDQTKFPIVNAELSFFGHAHLGTGDPYTPGFPSFNHTQFPPSRSSGLPNIPVQTISRAAAEKLFGNMEGDCPSDWKTDSTCRMVTSESKNVKLTVSNVLKEIKILNIFGVIKGFVEPDHYVVVGAQRDAWGPGAAKSGVGTALLLKLAQMFSDMVLKDGFQPSRSIIFASWSAGDFGSVGATEWLEGYLSSLHLKAFTYINLDKAVLGTSNFKVSASPLLYTLIEKTMQNVKHPVTGQFLYQDSNWASKVEKLTLDNAAFPFLAYSGIPAVSFCFCEDTDYPYLGTTMDTYKELIERIPELNKVARAAAEVAGQFVIKLTHDVELNLDYERYNSQLLSFVRDLNQYRADIKEMGLSLQWLYSARGDFFRATSRLTTDFGNAEKTDRFVMKKLNDRVMRVEYHFLSPYVSPKESPFRHVFWGSGSHTLPALLENLKLRKQNNGAFNETLFRNQLALATWTIQGAANALSGDVWDIDNEF',
    'gfap': 'MERRRITSAARRSYVSSGEMMVGGLAPGRRLGPGTRLSLARMPPPLPTRVDFSLAGALNAGFKETRASERAEMMELNDRFASYIEKVRFLEQQNKALAAELNQLRAKEPTKLADVYQAELRELRLRLDQLTANSARLEVERDNLAQDLATVRQKLQDETNLRLEAENNLAAYRQEADEATLARLDLERKIESLEEEIRFLRKIHEEEVRELQEQLARQQVHVELDVAKPDLTAALKEIRTQYEAMASSNMHEAEEWYRSKFADLTDAAARNAELLRQAKHEANDYRRQLQSLTCDLESLRGTNESLERQMREQEERHVREAASYQEALARLEEEGQSLKDEMARHLQEYQDLLNVKLALDIEIATYRKLLEGEENRITIPVQTFSNLQIRETSLDTKSVSEGHLKRNIVVKTVEMRDGEVIKESKQEHKDVM',
    'glp1': 'MAGAPGPLRLALLLLGMVGRAGPRPQGATVSLWETVQKWREYRRQCQRSLTEDPPPATDLFCNRTFDEYACWPDGEPGSFVNVSCPWYLPWASSVPQGHVYRFCTAEGLWLQKDNSSLPWRDLSECEESKRGERSSPEEQLLFLYIIYTVGYALSFSALVIASAILLGFRHLHCTRNYIHLNLFASFILRALSVFIKDAALKWMYSTAAQQHQWDGLLSYQDSLSCRLVFLLMQYCVAANYYWLLVEGVYLYTLLAFSVLSEQWIFRLYVSIGWGVPLLFVVPWGIVKYLYEDEGCWTRNSNMNYWLIIRLPILFAIGVNFLIFVRVICIVVSKLKANLMCKTDIKCRLAKSTLTLIPLLGTHEVIFAFVMDEHARGTLRFIKLFTELSFTSFQGLMVAILYCFVNNEVQLEFRKSWERWRLEHLHIQRDSSMKPLKCPTSSLSSGATAGSSMYTATCQASCS',
    'glast': 'MTKSNGEEPKMGGRMERFQQGVRKRTLLAKKKVQNITKEDVKSYLFRNAFVLLTVTAVIVGTILGFTLRPYRMSYREVKYFSFPGELLMRMLQMLVLPLIISSLVTGMAALDSKASGKMGMRAVVYYMTTTIIAVVIGIIIVIIIHPGKGTKENMHREGKIVRVTAADAFLDLIRNMFPPNLVEACFKQFKTNYEKRSFKVPIQANETLVGAVINNVSEAMETLTRITEELVPVPGSVNGVNALGLVVFSMCFGFVIGNMKEQGQALREFFDSLNEAIMRLVAVIMWYAPVGILFLIAGKIVEMEDMGVIGGQLAMYTVTVIVGLLIHAVIVLPLLYFLVTRKNPWVFIGGLLQALITALGTSSSSATLPITFKCLEENNGVDKRVTRFVLPVGATINMDGTALYEALAAIFIAQVNNFELNFGQIITISITATAASIGAAGIPQAGLVTMVIVLTSVGLPTDDITLIIAVDWFLDRLRTTTNVLGDSLGAGIVEHLSRHELKNRDVEMGNSVIEENEMKKPYQLIAQDNETEKPIDSETKM',
    'ncam': 'LQTKDLIWTLFFLGTAVSLQVDIVPSQGEISVGESKFFLCQVAGDAKDKDISWFSPNGEKLTPNQQRISVVWNDDSSSTLTIYNANIDDAGIYKCVVTGEDGSESEATVNVKIFQKLMFKNAPTPQEFREGEDAVIVCDVVSSLPPTIIWKHKGRDVILKKDVRFIVLSNNYLQIRGIKKTDEGTYRCEGRILARGEINFKDIQVIVNVPPTIQARQNIVNATANLGQSVTLVCDAEGFPEPTMSWTKDGEQIEQEEDDEKYIFSDDSSQLTIKKVDKNDEAEYICIAENKAGEQDATIHLKVFAKPKITYVENQTAMELEEQVTLTCEASGDPIPSITWRTSTRNISSEEKASWTRPEKQETLDGHMVVRSHARVSSLTLKSIQYTDAGEYICTASNTIGQDSQSMYLEVQYAPKLQGPVAVYTWEGNQVNITCEVFAYPSATISWFRDGQLLPSSNYSNIKIYNTPSASYLEVTPDSENDFGNYNCTAVNRIGQESLEFILVQADTPSSPSIDQVEPYSSTAQVQFDEPEATGGVPILKYKAEWRAVGEEVWHSKWYDAKEASMEGIVTIVGLKPETTYAVRLAALNGKGLGEISAASEF',
    'cereblon': 'MAGEGDQQDAAHNMGNHLPLLPAESEEEDEMEVEDQDSKEAKKPNIINFDTSLPTSHTYLGADMEEFHGRTLHDDDSCQVIPVLPQVMMILIPGQTLPLQLFHPQEVSMVRNLIQKDRTFAVLAYSNVQEREAQFGTTAEIYAYREEQDFGIEIVKVKAIGRQRFKVLELRTQSDGIQQAKVQILPECVLPSTMSAVQLESLNKCQIFPSKPVSREDQCSYKWWQKYQKRKFHCANLTSWPRWLYSLYDAETLMDRIKKQLREWDENLKDDSLPSNPIDFSYRVAACLPIDDVLRIQLLKIGSAIQRLRCELDIMNKCTSLCCKQCQETEITTKNEIFSLSLCGPMAAYVNPHGYVHETLTVYKACNLNLIGRPSTEHSWFPGYAWTVAQCKICASHIGWKFTATKKDMSPQKFWGLTRSALLPTIPDTEDEISPDKVILCL',
    'ligase': 'MASQPPEDTAESQASDELECKICYNRYNLKQRKPKVLECCHRVCAKCLYKIIDFGDSPQGVIVCPFCRFETCLPDDEVSSLPDDNNILVNLTCGGKGKKCLPENPTELLLTPKRLASLVSPSHTSSNCLVITIMEVQRESSPSLSSTPVVEFYRPASFDSVTTVSHNWTVWNCTSLLFQTSIRVLVWLLGLLYFSSLPLGIYLLVSKKVTLGVVFVSLVPSSLVILMVYGFCQCVCHEFLDCMAPPS',
    'skp2': 'MHRKHLQEIPDLSSNVATSFTWGWDSSKTSELLSGMGVSALEKEEPDSENIPQELLSNLGHPESPPRKRLKSKGSDKDFVIVRRPKLNRENFPGVSWDSLPDELLLGIFSCLCLPELLKVSGVCKRWYRLASDESLWQTLDLTGKNLHPDVTGRLLSQGVIAFRCPRSFMDQPLAEHFSPFRVQHMDLSNSVIEVSTLHGILSQCSKLQNLSLEGLRLSDPIVNTLAKNSNLVRLNLSGCSGFSEFALQTLLSSCSRLDELNLSWCFDFTEKHVQVAVAHVSETITQLNLSGYRKNLQKSDLSTLVRRCPNLVHLDLSDSVMLKNDCFQEFFQLNYLQHLSLSRCYDIIPETLLELGEIPTLKTLQVFGIVPDGTLQLLKEALPHLQINCSHFTTIARPTIGNKKNQEIWGIKCRLTLQKPSCL',
    'p53': 'MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD',
    'egfp': 'VSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK'
}

def save_logs_to_file(config, valid_fraction_log, score_logs, output_path):
    """
    Saves the logs to a CSV file.
    
    Parameters:
        valid_fraction_log (list): Log of valid fractions over iterations.
        score_logs (dict): Dict mapping score func names to lists of scores.
        output_path (str): Path to save the log CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    log_data = {
        "Iteration": list(range(1, len(valid_fraction_log) + 1)),
        "Valid Fraction": valid_fraction_log,
    }
    for name, log in score_logs.items():
        log_data[name] = log
        
    df = pd.DataFrame(log_data)
    
    # Save to CSV
    df.to_csv(output_path, index=False)

def plot_data(log1, log2=None, 
                    save_path=None, 
                    label1="Log 1", 
                    label2=None, 
                    title="Fraction of Valid Peptides Over Iterations", 
                    palette=None):
    """
    Plots one or two datasets with their mean values over iterations.

    Parameters:
        log1 (list): The first list of mean values for each iteration.
        log2 (list, optional): The second list of mean values for each iteration. Defaults to None.
        save_path (str): Path to save the plot. Defaults to None.
        label1 (str): Label for the first dataset. Defaults to "Log 1".
        label2 (str, optional): Label for the second dataset. Defaults to None.
        title (str): Title of the plot. Defaults to "Mean Values Over Iterations".
        palette (dict, optional): A dictionary defining custom colors for datasets. Defaults to None.
    """
    # Prepare data for log1
    data1 = pd.DataFrame({
        "Iteration": range(1, len(log1) + 1),
        "Fraction of Valid Peptides": log1,
        "Dataset": label1
    })

    # Prepare data for log2 if provided
    if log2 is not None:
        data2 = pd.DataFrame({
            "Iteration": range(1, len(log2) + 1),
            "Fraction of Valid Peptides": log2,
            "Dataset": label2
        })
        data = pd.concat([data1, data2], ignore_index=True)
    else:
        data = data1

    palette = {
        label1: "#8181ED",  # Default color for log1
        label2: "#D577FF"   # Default color for log2 (if provided)
    }

    # Set Seaborn theme
    sns.set_theme()
    sns.set_context("paper")

    # Create the plot
    sns.lineplot(
        data=data, 
        x="Iteration", 
        y="Fraction of Valid Peptides", 
        hue="Dataset", 
        style="Dataset", 
        markers=True, 
        dashes=False, 
        palette=palette
    )

    # Titles and labels
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Fraction of Valid Peptides")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()

def plot_data_with_distribution_seaborn(log1, log2=None, 
                                        save_path=None, 
                                        label1=None, 
                                        label2=None, 
                                        title=None):
    """
    Plots one or two datasets with the average values and distributions over iterations using Seaborn.

    Parameters:
        log1 (list of lists): The first list of scores (each element is a list of scores for an iteration).
        log2 (list of lists, optional): The second list of scores (each element is a list of scores for an iteration). Defaults to None.
        save_path (str): Path to save the plot. Defaults to None.
        label1 (str): Label for the first dataset. Defaults to "Fraction of Valid Peptide SMILES".
        label2 (str, optional): Label for the second dataset. Defaults to None.
        title (str): Title of the plot. Defaults to "Fraction of Valid Peptides Over Iterations".
    """
    # Prepare data for log1
    data1 = pd.DataFrame({
        "Iteration": np.repeat(range(1, len(log1) + 1), [len(scores) for scores in log1]),
        "Fraction of Valid Peptides": [float(score) for scores in log1 for score in scores],
        "Dataset": label1,
        "Style": "Log1"
    })

    # Prepare data for log2 if provided
    if log2 is not None:
        data2 = pd.DataFrame({
            "Iteration": np.repeat(range(1, len(log2) + 1), [len(scores) for scores in log2]),
            "Fraction of Valid Peptides": [float(score) for scores in log2 for score in scores],
            "Dataset": label2,
            "Style": "Log2"
        })
        data = pd.concat([data1, data2], ignore_index=True)
    else:
        data = data1
    
    palette = {
        label1: "#8181ED",  # Default color for log1
        label2: "#D577FF"   # Default color for log2 (if provided)
    }

    # Set Seaborn theme
    sns.set_theme()
    sns.set_context("paper")

    # Create the plot
    sns.relplot(
        data=data, 
        kind="line",
        x="Iteration", 
        y="Fraction of Valid Peptides", 
        hue="Dataset", 
        style="Style", 
        markers=True, 
        dashes=True,
        ci="sd",  # Show standard deviation
        height=5, 
        aspect=1.5,
        palette=palette
    )

    # Titles and labels
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Fraction of Valid Peptides")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()

def get_results_dir(config, prot_name):
    reliable = config.mcts.get("use_reliable_scores", False)

    folder = (
        "results_reliable_scores"
        if reliable else "results"
    )

    output_dir = os.path.join(
        config.base_path, folder, prot_name
    )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

@torch.no_grad()
def generate_valid_mcts(config, mdlm, prot1=None, prot2=None, filename=None, prot_name1=None, prot_name2 = None):
    tokenizer = mdlm.tokenizer
    max_sequence_length = config.sampling.seq_length
        
    # generate array of [MASK] tokens
    masked_array = mask_for_de_novo(config, max_sequence_length)

    inputs = tokenizer.encode(masked_array)
    
    inputs = {key: value.to(mdlm.device) for key, value in inputs.items()}
    
    # initialize root node
    rootNode = Node(config=config, tokens=inputs, timestep=0)
    # initalize tree search algorithm
    
    if config.mcts.perm:
        score_func_names = ['permeability', 'binding_affinity1', 'solubility', 'hemolysis', 'nonfouling']
        num_func = [0, 0, 0, 0, 0]
    elif config.mcts.dual:
        score_func_names = ['binding_affinity1', 'solubility', 'hemolysis', 'nonfouling', 'binding_affinity2']
        num_func = [0, 0, 0, 0, 0]
    elif config.mcts.single:
        if config.mode == 'binding':
            score_func_names = ['binding_affinity1']
        else:
            score_func_names = ['permeability']
        num_func = [0]
    else: 
        score_func_names = ['binding_affinity1', 'solubility', 'hemolysis', 'nonfouling']  
        num_func = [0, 0, 0, 0]
    

    # Remove unreliable objectives only when requested
    if config.mcts.get("use_reliable_scores", False):
        disabled_scores = {
            "binding_affinity1",
            "binding_affinity2",
            "solubility",
            "nonfouling",
        }

        score_func_names = [
            name for name in score_func_names
            if name not in disabled_scores
        ]

    if not score_func_names:
        raise ValueError(
            "No MCTS objectives remain after filtering."
        )

    # Match reward-vector dimensions to active objectives
    config.mcts.num_objectives = len(score_func_names)
    num_func = [0] * len(score_func_names)

    # Initialize root AFTER updating num_objectives
    rootNode = Node(
        config=config,
        tokens=inputs,
        timestep=0,
    )

    print("Active MCTS scores:", score_func_names)


    if not config.mcts.time_dependent:
        num_func = [0] * len(score_func_names)
    
    if prot1 and prot2 is not None:
        mcts = MCTS(config=config, max_sequence_length=max_sequence_length, mdlm=mdlm, score_func_names=score_func_names, prot_seqs=[prot1, prot2], num_func=num_func)
    elif prot1 is not None: 
        mcts = MCTS(config=config, max_sequence_length=max_sequence_length, mdlm=mdlm, score_func_names=score_func_names, prot_seqs=[prot1], num_func=num_func)
    elif config.mcts.single:
        mcts = MCTS(config=config, max_sequence_length=max_sequence_length, mdlm=mdlm, score_func_names=score_func_names, num_func=num_func)
    else:
        mcts = MCTS(config=config, max_sequence_length=max_sequence_length, mdlm=mdlm, score_func_names=score_func_names, num_func=num_func)

    paretoFront = mcts.forward(rootNode)

    output_dir = get_results_dir(config, prot_name1)

    output_log_path = os.path.join(
        output_dir, f"log_{filename}.csv"
    )
    save_logs_to_file(
        config,
        mcts.valid_fraction_log,
        mcts.score_logs,
        output_log_path,
    )

    plot_data(
        mcts.valid_fraction_log,
        save_path=os.path.join(
            output_dir, f"valid_{filename}.png"
        ),
    )

    for name in mcts.score_func_names:
        plot_data_with_distribution_seaborn(
            log1=mcts.score_logs[name],
            save_path=os.path.join(
                output_dir, f"{name}_{filename}.png"
            ),
            label1=f"Average {name}",
            title=f"Average {name} Over Iterations",
        )

    return paretoFront, inputs, score_func_names


@hydra.main(version_base=None, config_path='.', config_name='config')
def main(config):
    # Get parameters from config with defaults
    prot_name1 = config.get('prot_name1', 'gfap')
    prot_name2 = config.get('prot_name2', None)
    mode = config.get('mode', '2')
    model = config.get('model_type', 'mcts')
    length = config.get('length', '100')
    epoch = config.get('epoch', '7')
    
    filename = f'{mode}_{model}_length_{length}_epoch_{epoch}'

    tokenizer = SMILES_SPE_Tokenizer(f'{config.base_path}/src/tokenizer/new_vocab.txt', 
                                    f'{config.base_path}/src/tokenizer/new_splits.txt')
   
    mdlm = Diffusion.load_from_checkpoint(config.eval.checkpoint_path, config=config, tokenizer=tokenizer, strict=False, map_location='cpu', weights_only=False)
    
    mdlm.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    mdlm.to(device)
    

    print("loaded models...")
    analyzer = PeptideAnalyzer()
    
    # Look up protein sequences from names
    prot_seq1 = PROTEIN_SEQUENCES.get(prot_name1.lower(), LNR_SEQUENCES.get(prot_name1.lower()))
    prot_seq2 = PROTEIN_SEQUENCES.get(prot_name2.lower(), LNR_SEQUENCES.get(prot_name2.lower())) if prot_name2 else None
    
    if prot_seq1 is None:
        raise ValueError(f"Protein '{prot_name1}' not found in PROTEIN_SEQUENCES dictionary. Available proteins: {list(PROTEIN_SEQUENCES.keys())}")
    
    if prot_name2 and prot_seq2 is None:
        raise ValueError(f"Protein '{prot_name2}' not found in PROTEIN_SEQUENCES dictionary. Available proteins: {list(PROTEIN_SEQUENCES.keys())}")
    
    print(f"Using protein 1: {prot_name1}")
    if prot_name2:
        print(f"Using protein 2: {prot_name2}")
    
    t_start = time.time()
    paretoFront, input_array, score_func_names = generate_valid_mcts(config, mdlm, prot_seq1, prot_seq2, filename, prot_name1, prot_name2)

    score_labels = {
        "binding_affinity1": "Binding Affinity",
        "binding_affinity2": "Binding Affinity 2",
        "solubility": "Solubility",
        "hemolysis": "Hemolysis",
        "nonfouling": "Nonfouling",
        "permeability": "Permeability",
    }

    columns = [
        "Generated SMILES",
        "Perplexity",
        "Peptide Sequence",
    ] + [score_labels[name] for name in score_func_names]

    generation_results = []

    for sequence, v in paretoFront.items():
        generated_array = v["token_ids"].to(mdlm.device)

        perplexity = mdlm.compute_masked_perplexity(
            generated_array, input_array["input_ids"]
        )
        perplexity = round(perplexity, 4)

        aa_seq, seq_length = analyzer.analyze_structure(sequence)
        scores = list(v["scores"])

        generation_results.append(
            [sequence, perplexity, aa_seq] + scores
        )

        score_text = " | ".join(
            f"{score_labels[name]}: {score}"
            for name, score in zip(score_func_names, scores)
        )

        print(
            f"perplexity: {perplexity} | "
            f"length: {seq_length} | "
            f"smiles sequence: {sequence} | "
            f"amino acid sequence: {aa_seq} | "
            f"{score_text}"
        )
        sys.stdout.flush()

    df = pd.DataFrame(generation_results, columns=columns)

    output_dir = get_results_dir(config, prot_name1)
    df.to_csv(
        os.path.join(output_dir, f"{filename}.csv"),
        index=False,
    )

    # ── timing ──
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Generation complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Pareto front size: {len(df)}")

    # ── score statistics ──
    score_cols = [c for c in df.columns if c not in ('Generated SMILES', 'Peptide Sequence')]
    print(f"\n{'Score':<22} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print('-' * 58)
    for col in score_cols:
        vals = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(vals) == 0:
            continue
        print(f"{col:<22} {vals.mean():8.4f} {vals.std():8.4f} {vals.min():8.4f} {vals.max():8.4f}")
    print('=' * 60)


if __name__ == "__main__":
    main()
