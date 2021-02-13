"""Functions for creating and scoring CATH datasets"""

import numpy as np
import pandas as pd
import ampal
import gzip
import glob
import subprocess
import multiprocessing
import os
from pathlib import Path
from sklearn import metrics
from benchmark import config
import string
from subprocess import CalledProcessError
import re

def read_data(CATH_file: str) -> pd.DataFrame:
    """If CATH .csv exists, loads the DataFrame. If CATH .txt exists, makes DataFrame and saves it.

    Parameters
    ----------
    CATH_file: str
        PATH to CATH .txt file.

    Returns
    -------
    DataFrame containing CATH and PDB codes."""
    path=Path(CATH_file)
    #load .csv if exists, faster than reading .txt
    if path.with_suffix('.csv').exists():
        df = pd.read_csv(path.with_suffix('.csv'), index_col=0)
        # start, stop needs to be str
        df["start"] = df["start"].apply(str)
        df["stop"] = df["stop"].apply(str)
        return df

    else:
        cath_info = []
        temp = []
        start_stop = []
        # ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/cath-classification-data/
        with open(path) as file:
            for line in file:
                if line[:6] == "DOMAIN":
                    # PDB
                    temp.append(line[10:14])
                    # chain
                    temp.append(line[14])
                if line[:6] == "CATHCO":
                    # class, architecture, topology, homologous superfamily
                    cath = [int(i) for i in line[10:].strip("\n").split(".")]
                    temp = temp + cath
                if line[:6] == "SRANGE":
                    j = line.split()
                    # start and stop resi, can be multiple for the same chain
                    # must be str to deal with insertions (1A,1B) later.
                    start_stop.append([str(j[1][6:]), str(j[2][5:])])
                if line[:2] == "//":
                    # keep fragments from the same chain as separate entries
                    for fragment in start_stop:
                        cath_info.append(temp + fragment)
                    start_stop = []
                    temp = []
        df = pd.DataFrame(
            cath_info,
            columns=[
                "PDB",
                "chain",
                "class",
                "architecture",
                "topology",
                "hsf",
                "start",
                "stop",
            ],
        )
        df.to_csv(path.with_suffix('.csv'))
        return df

def tag_dssp_data(assembly: ampal.Assembly):
    """Same as ampal.dssp.tag_dssp_data(), but fixed a bug with insertions. Tags each residue in ampal.Assembly with secondary structure.

    Parameters
    ----------
    assembly: ampal.Assembly
        Protein assembly."""

    dssp_out = ampal.dssp.run_dssp(assembly.pdb, path=False)
    dssp_data = ampal.dssp.extract_all_ss_dssp(dssp_out, path=False)
    for i,record in enumerate(dssp_data):
        rnum, sstype, chid, _, phi, psi, sacc = record
        #deal with insertions
        if len(chid)>1:
            for i,res in enumerate(assembly[chid[1]]):
                if res.insertion_code==chid[0] and assembly[chid[1]][i].tags=={}:
                    assembly[chid[1]][i].tags['dssp_data'] = {
                    'ss_definition': sstype,
                    'solvent_accessibility': sacc,
                    'phi': phi,
                    'psi': psi
                    }
                    break
                
        else:
            assembly[chid][str(rnum)].tags['dssp_data'] = {
            'ss_definition': sstype,
            'solvent_accessibility': sacc,
            'phi': phi,
            'psi': psi
            }

def get_sequence(series: pd.Series) -> str:
    """Gets a sequence of from PDB file, CATH fragment indexes and secondary structure labels.

    Parameters
    ----------
    series: pd.Series
        Series containing one CATH instance.
    path:str
        Path to PDB dataset directory.

    Returns
    -------
    If PDB exists, returns sequence, dssp sequence, and start and stop index for CATH fragment. If not, returns np.NaN

    Notes
    -----
    Unnatural amino acids are removed"""
    
    #path=config.PATH_TO_PDB/series.PDB[1:3]/f"pdb{series.PDB}.ent.gz"
    path=config.PATH_TO_ASSEMBLIES/series.PDB[1:3]/f"{series.PDB}.pdb1.gz"
            
    if path.exists():
        with gzip.open(path,"rb") as protein:
            assembly = ampal.load_pdb(protein.read().decode(), path=False)
            #check is assembly has multiple states, pick the first
            if isinstance(assembly,ampal.assembly.AmpalContainer):
                if assembly[0].id.count('_state_')>0:
                    assembly=assembly[0]
            # convert pdb res id into sequence index,
            # some files have discontinuous residue ids so ampal.get_slice_from_res_id() does not work
            # convert pdb res id into sequence index,
            # some files have discontinuous residue ids so ampal.get_slice_from_res_id() does not work
            start = 0
            stop = 0
            # if nmr structure, get 1st model
            if isinstance(assembly, ampal.AmpalContainer):
                assembly = assembly[0]
            #run dssp
            try:
                tag_dssp_data(assembly)
            except CalledProcessError:
                print(f"dssp failed on {series.PDB}.pdb1.")
                return np.NaN, np.NaN, np.NaN, np.NaN, np.NaN
            #some biological assemblies are broken
            try:
                chain = assembly[series.chain]
            except KeyError:
                print(f"{series.PDB}.pdb1 is missing chain {series.chain}.")
                return np.NaN, np.NaN, np.NaN, np.NaN, np.NaN

            #compatibility with evoef and leo's model, store uncommon residue index in a separate column and include regular amino acid in the sequence
            sequence=''
            uncommon_index=[]
            dssp=''
            for i, residue in enumerate(chain):
                #add dssp data, assume random structure if dssp did not return anything for this residue
                try:
                    dssp += residue.tags['dssp_data']['ss_definition']
                except KeyError:
                    dssp+=' '
                #deal with uncommon residues
                one_letter_code=ampal.amino_acids.get_aa_letter(residue.mol_code)
                if one_letter_code=='X':
                    try:
                        uncommon_index.append(i)
                        sequence+=ampal.amino_acids.get_aa_letter(config.UNCOMMON_RESIDUE_DICT[residue.mol_code])
                    except KeyError:
                        print(f"{series.PDB}.pdb1 has unrecognized amino acid {residue.mol_code}.")
                        return np.NaN, np.NaN, np.NaN, np.NaN, np.NaN
                else:
                    sequence+=one_letter_code
                    
                # deal with insertions
                if series.start[-1].isalpha():
                    if (residue.id + residue.insertion_code) == series.start:
                        start = i
                else:
                    if residue.id == series.start:
                        start = i
                if series.stop[-1].isalpha():
                    if (residue.id + residue.insertion_code) == series.stop:
                        stop = i
                else:
                    if residue.id == series.stop:
                        stop = i
       
        return sequence, dssp, start, stop, uncommon_index
    else:
        print(f"{series.PDB}.pdb1 is missing.")
        return np.NaN, np.NaN, np.NaN, np.NaN, np.NaN

def get_pdbs(
    df: pd.DataFrame, cls: int, arch: int = 0, topo: int = 0, homologous_sf: int = 0
) -> pd.DataFrame:
    """Gets PDBs based on CATH code, at least class has to be specified.

    Parameters
    ----------
        df: pd.DataFrame
            DataFrame containing CATH dataset.
        cls: int
            CATH class
        arch: int = 0
            CATH architecture
        topo: int = 0
            CATH topology
        homologous_sf: int = 0
            CATH homologous superfamily

    Returns
    -------
    DataFrame containing PDBs with specified CATH code."""

    if homologous_sf != 0:
        return df.loc[
            (df["class"] == cls)
            & (df["topology"] == topo)
            & (df["architecture"] == arch)
            & (df["hsf"] == homologous_sf)
        ].copy()
    elif topo != 0:
        return df.loc[
            (df["class"] == cls)
            & (df["topology"] == topo)
            & (df["architecture"] == arch)
        ].copy()
    elif arch != 0:
        return df.loc[(df["class"] == cls) & (df["architecture"] == arch)].copy()
    else:
        return df.loc[(df["class"] == cls)].copy()


def append_sequence(df: pd.DataFrame) -> pd.DataFrame:
    """Get sequences for all entries in the dataframe, changes start and stop from PDB resid to index number.

    Parameters
    ----------
    df: pd.DataFrame
        CATH dataframe

    Returns
    -------
    DataFrame with existing sequences"""
    working_copy = df.copy()

    (
        working_copy.loc[:, "sequence"],
        working_copy.loc[:, "dssp"],
        working_copy.loc[:, "start"],
        working_copy.loc[:, "stop"],
         working_copy.loc[:, "uncommon_index"],
    ) = zip(*[get_sequence(x) for i, x in df.iterrows()])
    # remove missing entries
    working_copy.dropna(inplace=True)
    # change index from float to int
    working_copy.loc[:, "start"] = working_copy["start"].apply(int)
    working_copy.loc[:, "stop"] = working_copy["stop"].apply(int)
    return working_copy

def filter_with_TS50(df: pd.DataFrame) -> pd.DataFrame:
    """Takes CATH datarame and returns PDB chains from TS50 dataset

    Parameters
    ----------
    df: pd.DataFrame
        CATH DataFrame

    Returns
    -------
    TS50 DataFrame

    Reference
    ----------
     https://doi.org/10.1002/prot.25868 (ProDCoNN)"""

    frame_copy = df.copy()
    frame_copy["PDB+chain"] = df.PDB + df.chain
    # must be upper letters for string comparison
    frame_copy["PDB+chain"] = frame_copy["PDB+chain"].str.upper()
    return df.loc[frame_copy["PDB+chain"].isin(config.ts50)]
    
def filter_with_user_list(df: pd.DataFrame, path: str, ispisces:bool = False)->pd.DataFrame:
    """Selects PDB chains specified in .txt file.
    Parameters
    ----------
    df: pd.DataFrame
        CATH info containing dataframe
    path: str
        Path to .txt file
    ispisces:bool = False
        Reads pisces formating if True, otherwise pdb+chain, e.g., 1a2bA\n.
    
    Returns
    -------
    DataFrame with selected chains,duplicates are removed."""
    path=Path(path)
    with open(path) as file:
        if ispisces:
            filtr = [x.split()[0] for x in file.readlines()[1:]]
        else:
            filtr=[x.upper().strip('\n') for x in file.readlines()]
    frame_copy = df.copy()
    frame_copy["PDB+chain"] = df.PDB + df.chain
    # must be upper letters for string comparison
    frame_copy["PDB+chain"] = frame_copy["PDB+chain"].str.upper()
    return df.loc[frame_copy["PDB+chain"].isin(filtr)].drop_duplicates(subset=['PDB','chain'])

def lookup_blosum62(res_true: str, res_prediction: str) -> int:
    """Returns score from the matrix.

    Parameters
    ----------
    a: str
        First residue code.
    b: str
        Second residue code.

    Returns
    --------
    Score from the matrix."""

    if (res_true, res_prediction) in config.blosum62.keys():
      return config.blosum62[res_true, res_prediction] 
    else:
      return config.blosum62[res_prediction, res_true]

def secondary_score(true_seq: np.array, predicted_seq: np.array, dssp: str, issequence:bool) -> list:
    """Calculates sequence recovery rate for each secondary structure type.

    Parameters
    ----------
    true_seq: str,
        True sequence.
    predicted_seq: str
        Predicted sequence.
    dssp: str
        string with dssp results
    issequence: bool
        True if sequence, false if probability matrix.

    Returns
    -------
    List with sequence recovery for helices, beta sheets, random coils and structured loops"""

    true=[[],[],[],[]]
    prediction=[[],[],[],[]]
    for structure,truth,pred in zip(dssp,true_seq,predicted_seq):
        if structure=="H" or structure=="I" or structure=="G":
            true[0].append(truth)
            prediction[0].append(pred)
        elif structure=='E':
            true[1].append(truth)
            prediction[1].append(pred)
        elif structure=="B" or structure=="T" or structure=="S":
            true[2].append(truth)
            prediction[2].append(pred)
        else:
            true[3].append(truth)
            prediction[3].append(pred)
        results=[]
    for seq_type in range(len(true)):
        if issequence:
            if len(true[seq_type])>0:
                 results+=[metrics.accuracy_score(true[seq_type],prediction[seq_type]),metrics.recall_score(true[seq_type],prediction[seq_type],average='macro')]
            else:
                results+=[np.NaN,np.NaN]
        else:
            if len(true[seq_type])>0:
                seq_prediction=list(most_likely_sequence(prediction[seq_type]))
                results+=[metrics.accuracy_score(true[seq_type],seq_prediction),metrics.recall_score(true[seq_type],seq_prediction,average='macro'),metrics.top_k_accuracy_score(true[seq_type],prediction[seq_type],k=3,labels=config.acids)]
            else:
                results+=[np.NaN,np.NaN,np.NaN]
    return results

def run_Evo2EF(pdb: str, chain: str, number_of_runs: str, working_dir: Path):
    """Runs a shell script to predict sequence with EvoEF2

    Patameters
    ----------
    path: str
        Path to PDB biological unit.
    pdb: str
        PDB code.
    chain: str
        Chain code.
    number_of_runs: str
       Number of sequences to be generated.
    working_dir: str
      Dir where to store temporary files and results

    Returns
    -------
    Nothing."""

    #evo.sh must be in the same directory as this file.
    p = subprocess.Popen(
        [
            os.path.dirname(os.path.realpath(__file__))+"/evo.sh",
            pdb,
            chain,
            number_of_runs,
            working_dir,
        ]
    )
    p.wait()
    print(f"{pdb}{chain} done.")


def multi_Evo2EF(df: pd.DataFrame, number_of_runs: int, working_dir: str, max_processes: int = 8):
    """Runs Evo2EF on all PDB chains in the DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with PDB and chain codes.
    number_of_runs: int
        Number of sequences to be generated for each PDB file.
    max_processes: int = 8
        Number of cores to use, default is 8.
    working_dir: str
      Dir where to store temporary files and results.
      
    Returns
    --------
    Nothing."""

    inputs = []
    # remove duplicated chains
    df = df.drop_duplicates(subset=["PDB", "chain"])

    #check if working directory exists. Make one if doesn't exist.
    working_dir=Path(working_dir)
    if not working_dir.exists():
      os.makedirs(working_dir)
    if not (working_dir/'results').exists():
      os.makedirs(working_dir/'/results')

    for i, protein in df.iterrows():
        with gzip.open(config.PATH_TO_ASSEMBLIES/protein.PDB[1:3]/f"{protein.PDB}.pdb1.gz") as file:
            assembly = ampal.load_pdb(file.read().decode(), path=False)
        #fuse all states of the assembly into one state to avoid EvoEF2 errors.
        empty_polymer=ampal.Assembly()
        chain_id=[]
        for polymer in assembly:
            for chain in polymer:
                empty_polymer.append(chain)
                chain_id.append(chain.id)
        #relabel chains to avoid repetition
        str_list=string.ascii_uppercase.replace(protein.chain, "")
        index=chain_id.index(protein.chain)
        chain_id=list(str_list[:len(chain_id)])
        chain_id[index]=protein.chain
        empty_polymer.relabel_polymers(chain_id)
        pdb_text=empty_polymer.make_pdb(alt_states=False,ligands=False)
        #writing new pdb with AMPAL fixes most of the errors with EvoEF2.
        with open((working_dir/protein.PDB).with_suffix(".pdb1"),'w') as pdb_file:
            pdb_file.write(pdb_text)
        inputs.append((protein.PDB, protein.chain, str(number_of_runs),working_dir))

    with multiprocessing.Pool(max_processes) as P:
        P.starmap(run_Evo2EF, inputs)


def load_prediction_sequence(df: pd.DataFrame,path:str) -> pd.DataFrame:
    """Loads predicted sequences from .txt to dictionary, drops entries for which sequence prediction fails.
        Parameters
        ----------
        df: pd.DataFrame
            CATH dataframe
        path:str
            Path to prediction directory.
        
        Returns
        -------
        Dictionary with predicted sequences."""

    predicted_sequences = {}
    path=Path(path)
    for i, protein in df.iterrows():
        prediction_path = path/f"{protein.PDB}{protein.chain}.txt"
        # check for empty and missing files
        if prediction_path.exists() and os.path.getsize(prediction_path)>0:
            with open(prediction_path) as prediction:
                seq = prediction.readline().split()[0]
                if seq != '0':
                    predicted_sequences[protein.PDB+protein.chain]=seq
                else:
                    print(
                        f"{protein.PDB}{protein.chain} prediction does not exits, EvoEF2 returned 0."
                    )
        else:
            print(f"{protein.PDB}{protein.chain} prediction does not exits.")
    return predicted_sequences

def load_prediction_matrix(df:pd.DataFrame,path_to_dataset: str, path_to_probabilities:str)->dict:
    """Loads predicted probabilities from .csv file to dictionary, drops entries for which sequence prediction fails.
        Parameters
        ----------
        df: pd.DataFrame
            CATH dataframe
        path_to_dataset: str
            Path to prediction dataset labels.
        path_to_probabilities:str
            Path to .csv file with probabilities.
        
        Returns
        -------
        Dictionary with predicted sequences."""

    path_to_dataset=Path(path_to_dataset)
    path_to_probabilities=Path(path_to_probabilities)
    with open(path_to_dataset) as file:
        labels=[(x.split(',')[0],int(x.split(',')[2])) for x in file.readlines()]
    predictions=pd.read_csv(path_to_probabilities,header=None).values
    empty_dict={k:[] for k in df.PDB.values+df.chain.values}
    for probability,pdb in zip(predictions,labels):
        if pdb[0] in empty_dict:
            empty_dict[pdb[0]].append((pdb[1],probability))
    #sort predictions into a protein sequence
    for protein in empty_dict:
        empty_dict[protein]=np.array([x[1] for x in sorted(empty_dict[protein])])
    #drop keys with missing values
    empty_dict={key:empty_dict[key] for key in empty_dict if len(empty_dict[key])!=0}
    return empty_dict

def most_likely_sequence(probability_matrix) -> str:
    if len(probability_matrix)>0:
        most_likely_seq=[config.acids[x] for x in np.argmax(probability_matrix, axis=1)]
        return ''.join(most_likely_seq)
    else:
        return ''

def score(
    df: pd.DataFrame, predictions:dict, by_fragment: bool=True, ignore_uncommon=False,score_sequence=False) -> list:
    """Concatenates and scores all predicted sequences in the DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with CATH fragment info. The frame must have predicted sequence, true sequence and start/stop index of CATH fragment.
    predictions: dict
        Dictionary with loaded predictions.
    by_fragment: bool
        If true scores only CATH fragments, if False, scores entire chain.
    ignore_uncommon=True
        If True, ignores uncommon residues in accuracy calculations.
    score_sequence=False
        True if dictionary contains sequences, False if probability matrices(matrix shape n,20).

    Returns
    --------
    A list with sequence recovery, similarity, f1 and secondary structure scores"""

    sequence=''
    dssp=''
    if score_sequence:
        prediction=''
    else:
        prediction=np.empty([0,20])
    for i,protein in df.iterrows():
        if protein.PDB+protein.chain in predictions: 
            start=protein.start
            stop=protein.stop
            predicted_sequence=predictions[protein.PDB+protein.chain]

            #remove uncommon acids
            if ignore_uncommon and protein.uncommon_index!=[]:
                protein_sequence=''.join([x for i,x in enumerate(protein.sequence) if i not in protein.uncommon_index])
                protein_dssp=''.join([x for i,x in enumerate(protein.dssp) if i not in protein.uncommon_index])
                #update start and stop indexes
                start=start-(np.array(protein.uncommon_index)<=start).sum()
                stop=stop-(np.array(protein.uncommon_index)<=stop).sum()
            else:
                protein_sequence=protein.sequence
                protein_dssp=protein.dssp

            #check length
            if len(protein_sequence)!=len(predicted_sequence):
                #prediction is multimer
                if len(predicted_sequence)%len(protein_sequence)==0:
                    predicted_sequence=predicted_sequence[0:len(protein_sequence)]
                else:
                    print(f"{protein.PDB}{protein.chain} sequence, predicted sequence and dssp length do not match.")
                    continue

            if by_fragment:
                protein_sequence=protein_sequence[start:stop+1]
                protein_dssp=protein_dssp[start:stop+1]
                predicted_sequence=predicted_sequence[start:stop+1]

            if len(protein_sequence)==len(predicted_sequence) and len(protein_sequence)==len(protein_dssp):
                sequence+=protein_sequence
                dssp+=protein_dssp
                if score_sequence:
                    prediction+=predicted_sequence    
                else:
                    prediction=np.concatenate([prediction,predicted_sequence],axis=0)
            else:
                print(f"{protein.PDB}{protein.chain} sequence, predicted sequence and dssp length do not match.")

    sequence=np.array(list(sequence))
    dssp=np.array(list(dssp))
    if score_sequence:        
        prediction=np.array(list(prediction))
        sequence_recovery=metrics.accuracy_score(sequence,prediction)
        recall=metrics.recall_score(sequence,prediction,average='macro')
        similarity_score=[1 if lookup_blosum62(a,b)>0 else 0 for a,b in zip(sequence,prediction)]
        similarity_score=sum(similarity_score)/len(similarity_score)
        alpha,alpha_recall,beta,beta_recall,loop,loop_recall,random,random_recall=secondary_score(prediction,sequence,dssp,True)
        return sequence_recovery,similarity_score,recall,alpha,alpha_recall,beta,beta_recall,loop,loop_recall,random,random_recall
    
    else:
        most_likely_seq=np.array(list(most_likely_sequence(prediction)))
        sequence_recovery=metrics.accuracy_score(sequence,most_likely_seq)
        recall=metrics.recall_score(sequence,most_likely_seq,average='macro')
        similarity_score=[1 if lookup_blosum62(a,b)>0 else 0 for a,b in zip(sequence,most_likely_seq)]
        similarity_score=sum(similarity_score)/len(similarity_score)
        top_three_score=metrics.top_k_accuracy_score(sequence,prediction,k=3,labels=config.acids)

        alpha,alpha_recall,alpha_three,beta,beta_recall,beta_three,loop,loop_recall,loop_three,random,random_recall,random_three=secondary_score(sequence,prediction,dssp,False)
        return sequence_recovery,top_three_score,similarity_score,recall,alpha,alpha_recall,alpha_three,beta,beta_recall,beta_three,loop,loop_recall,loop_three,random,random_recall,random_three


def score_by_architecture(df:pd.DataFrame,predictions:dict,by_fragment: bool=True,ignore_uncommon=False,score_sequence=False)->pd.DataFrame:
    """Groups the predictions by architecture and scores each separately.

        Parameters
        ----------
        df:pd.DataFrame
            DataFrame containing predictions, cath codes and true sequences.
        by_fragment: bool =True
            If true scores only CATH fragments, if False, scores entire chain.
        
        Returns
        -------
        DataFrame with accuracy, similarity, f1, and secondary structure accuracy for each architecture type and a dictionary with overal metrics."""

    architectures=df.drop_duplicates(subset=['class','architecture'])['architecture'].values
    classes=df.drop_duplicates(subset=['class','architecture'])['class'].values
    scores=[]
    names=[]
    for cls,arch in zip(classes,architectures):
        scores.append(score(get_pdbs(df,cls,arch),predictions,by_fragment,ignore_uncommon,score_sequence))
        #lookup normal names
        names.append(config.architectures[f"{cls}.{arch}"])
    if score_sequence:
        score_frame=pd.DataFrame(scores,columns=['accuracy','similarity','recall','alpha','alpha_recall','beta','beta_recall','loops','loops_recall','random','random_recall'],index=[classes,architectures])
        general_dict={x:y for x,y in zip(['accuracy','similarity','recall','alpha','alpha_recall','beta','beta_recall','loops','loops_recall','random','random_recall'],score(df,predictions,by_fragment,ignore_uncommon,score_sequence))}
    else:
        score_frame=pd.DataFrame(scores,columns=['accuracy','top3_accuracy','similarity','recall','alpha','alpha_recall','alpha_three','beta','beta_recall','beta_three','loops','loops_recall','loops_three','random','random_recall','random_three'],index=[classes,architectures])
        general_dict={x:y for x,y in zip(['accuracy','top3_accuracy','similarity','recall','alpha','alpha_recall','alpha_three','beta','beta_recall','beta_three','loops','loops_recall','loops_three','random','random_recall','random_three'],score(df,predictions,by_fragment,ignore_uncommon,score_sequence))}  
    #get meaningful names
    score_frame['name']=names
    return score_frame,general_dict

def score_each(df: pd.DataFrame, predictions:dict, by_fragment: bool=True, ignore_uncommon=False,score_sequence=False) -> list:
    """Calculates accuracy and recall for each protein in DataFrame separately.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with CATH fragment info. The frame must have predicted sequence, true sequence and start/stop index of CATH fragment.
    predictions: dict
        Dictionary with loaded predictions.
    by_fragment: bool
        If true scores only CATH fragments, if False, scores entire chain.
    ignore_uncommon=True
        If True, ignores uncommon residues in accuracy calculations.
    score_sequence=False
        True if dictionary contains sequences, False if probability matrices(matrix shape n,20).

    Returns
    --------
    A list with sequence recovery and recall for each protein in DataFrame"""
    accuracy=[]
    recall=[]
    for i,protein in df.iterrows():
        if protein.PDB+protein.chain in predictions: 
            start=protein.start
            stop=protein.stop
            predicted_sequence=predictions[protein.PDB+protein.chain]

            #remove uncommon acids
            if ignore_uncommon and protein.uncommon_index!=[]:
                protein_sequence=''.join([x for i,x in enumerate(protein.sequence) if i not in protein.uncommon_index])
                start=start-(np.array(protein.uncommon_index)<=start).sum()
                stop=stop-(np.array(protein.uncommon_index)<=stop).sum()
            else:
                protein_sequence=protein.sequence

            #check length
            if len(protein_sequence)!=len(predicted_sequence):
                #prediction is multimer
                if len(predicted_sequence)%len(protein_sequence)==0:
                    predicted_sequence=predicted_sequence[0:len(protein_sequence)]
                else:
                    print(f"{protein.PDB}{protein.chain} sequence, predicted sequence and dssp length do not match.")
                    accuracy.append(np.NaN)
                    recall.append(np.NaN)
                    continue
            if by_fragment:
                protein_sequence=protein_sequence[start:stop+1]
                predicted_sequence=predicted_sequence[start:stop+1]
            if score_sequence:
                accuracy.append(metrics.accuracy_score(list(protein_sequence),list(predicted_sequence)))
                recall.append(metrics.recall_score(list(protein_sequence),list(predicted_sequence),average='macro'))
            else:
                accuracy.append(metrics.accuracy_score(list(protein_sequence),list(most_likely_sequence(predicted_sequence))))
                recall.append(metrics.recall_score(list(protein_sequence),list(most_likely_sequence(predicted_sequence)),average='macro'))
        else:
            accuracy.append(np.NaN)
            recall.append(np.NaN)
            
    return accuracy,recall       

def get_resolution(df:pd.DataFrame)->list:
    """Gets resolution of each structure in DataFrame

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with CATH fragment info.
    
    Returns
    -------
    List with resolutions."""

    res=[]
    for i,protein in df.iterrows():
        path=config.PATH_TO_PDB/protein.PDB[1:3]/f"pdb{protein.PDB}.ent.gz"
            
        if path.exists():
            with gzip.open(path,"rb") as pdb:
                pdb_text = pdb.read().decode()
            item=re.findall("REMARK   2 RESOLUTION.*$",pdb_text,re.MULTILINE)
            res.append(float(item[0].split()[3]))
        else:
            res.append(np.NaN)
    return res
