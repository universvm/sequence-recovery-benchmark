import pandas as pd
from benchmark import config
import ampal
from benchmark import get_cath
import gzip
from pathlib import Path
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

def _annotate_ampalobj_with_data_tag(
    ampal_structure,
    data_to_annotate,
    tags,
) -> ampal.assembly:
    """
    Assigns a data point to each residue equivalent to the prediction the
    tag value. The original value of the tag will be reset to the minimum value
    to allow for a more realistic color comparison.
    Parameters
    ----------
    ampal_structure : ampal.Assembly or ampal.AmpalContainer
        Ampal structure to be modified. If an ampal.AmpalContainer is passed,
        this will take the first Assembly in the ampal.AmpalContainer `ampal_structure[0]`.
    data_to_annotate : numpy.ndarray of numpy.ndarray of floats
        Numpy array with data points to annotate (x, n) where x is the
        numer of arrays with data points (eg, [ entropy, accuracy ] ,
        x = 2n) and n is the number of residues in the structure.
    tags : t.List[str]
        List of string tags of the pdb object (eg. "b-factor")
    Returns
    -------
    ampal_structure : Assembly
        Ampal structure with modified B-factor values.

    Notes
    -----
    Same as _annotate_ampalobj_with_data_tag from TIMED but can deal with missing unnatural amino acids for compatibility with EvoEF2."""

    assert len(tags) == len(
        data_to_annotate
    ), "The number of tags to annotate and the type of data to annotate have different lengths."
    # Deals with structures from NMR as ampal returns Container of Assemblies
    if isinstance(ampal_structure, ampal.AmpalContainer):
        warnings.warn(
            f"Selecting the first state from the NMR structure {ampal_structure.id}"
        )
        ampal_structure = ampal_structure[0]

    if len(data_to_annotate) > 1:
        assert len(data_to_annotate[0]) == len(data_to_annotate[1]), (
            f"Data to annotatate has shape {len(data_to_annotate[0])} and "
            f"{len(data_to_annotate[1])}. They should be the same."
        )

    for i, tag in enumerate(tags):
        # Reset existing values:
        for atom in ampal_structure.get_atoms(ligands=True, inc_alt_states=True):
            atom.tags[tag] = np.min(data_to_annotate[i])

    # Apply data as tag:
    for chain in ampal_structure:
        for i, tag in enumerate(tags):

            # Check if chain is Polypeptide (it might be DNA for example...)
            if isinstance(chain, ampal.Polypeptide):
                if len(chain) != len(data_to_annotate[i]):
                    #EvoEF2 predictions drop uncommon amino acids
                    if len(chain)-chain.sequence.count('X')==len(data_to_annotate[i]):
                        for residue in chain:
                            counter=0
                            if ampal.amino_acids.get_aa_letter(residue)=='X':
                                continue
                            else:
                                for atom in residue:
                                    atom.tags[tag] =  data_to_annotate[i][counter]
                                    counter+=1
                    else:
                        print('Length is not equal')
                        return  
                for residue, data_val in zip(chain, data_to_annotate[i]):
                    for atom in residue:
                        atom.tags[tag] = data_val

    return ampal_structure

def show_accuracy(df:pd.DataFrame, pdb:str, output:str):
    accuracy=[]
    sequence=df[df.PDB==pdb].sequence.values[0]
    predictions=df[df.PDB==pdb].predicted_sequences.values[0]
    for resa,resb in zip(sequence,predictions):
        #correct predictions are given constant score so they stand out in the figure.
        #e.g., spectrum b, blue_white_red, maximum=6,minimum=-6 gives nice plots. Bright red shows correct predictions
        #Red shades indicate substitutions with positive score, white=0, blue shades show substiutions with negative score.
        if resa==resb:
            accuracy.append(6)
        #incorrect predictions will be coloured by blossum62 score.
        else:
            accuracy.append(get_cath.lookup_blosum62(resa,resb))
    path_to_protein=config.PATH_TO_ASSEMBLIES/pdb[1:3]/f"pdb{pdb}.ent.gz"
    with gzip.open(path_to_protein, "rb") as protein:
        assembly = ampal.load_pdb(protein.read().decode(), path=False)
    ##add entropy in the future
    curr_annotated_structure = _annotate_ampalobj_with_data_tag(assembly,[accuracy],tags=["bfactor"])
    with open(output, "w") as f:
        f.write(curr_annotated_structure.pdb)

def compare_model_accuracy(models:list,name:str,model_labels=list):
    #plot maximum 9 models, otherwise the plot is a complete mess
    if len(models)>8:
        models=models[0:9]
    colors=sns.color_palette()
    #combine 4 and 6 to make plots nicer. Works with any number of CATH classes.
    class_key=[x[0] for x in models[0].index]
    class_key=list(dict.fromkeys(class_key))
    if 4 in class_key and 6 in class_key:
        class_key=[x for x in class_key if x!=4 and x!=6]
        class_key.append([4,6])
    ratios=[models[0].loc[class_key[i]].shape[0] for i in range(len(class_key))]
    fig, ax = plt.subplots(2,len(class_key),figsize=(12*len(class_key),10), gridspec_kw={'width_ratios': ratios},squeeze=False)
    for i in range(len(class_key)):
        index=np.arange(0,models[0].loc[class_key[i]].shape[0])
        for j,frame in enumerate(models):  
            value_accuracy=frame.loc[class_key[i]].accuracy.values
            value_recall=frame.loc[class_key[i]].recall.values
            ax[0][i].bar(x=index+j*0.1, height=value_accuracy, width=0.1, align='center', color=colors[j],label=model_labels[j])
            #show top3 accuracy if it exists
            if 'top3_accuracy' in frame:
                value_top_three=frame.loc[class_key[i]].top3_accuracy.values
                ax[0][i].scatter(x=index+j*0.1, y=value_top_three,marker="_", s=50, color=colors[j])
                ax[0][i].vlines(x=index+j*0.1, ymin=0, ymax=value_top_three, color=colors[j], linewidth=2)
            ax[1][i].bar(x=index+j*0.1, height=value_recall, width=0.1, align='center', color=colors[j])
# Title, Label, Ticks and Ylim
        ax[0][i].set_title(config.classes[i+1], fontdict={'size':22})
        ax[1][i].set_title(config.classes[i+1], fontdict={'size':22})
        ax[0][i].set_ylabel('Accuracy')
        ax[1][i].set_ylabel('Recall')
        ax[0][i].set_xticks(index)
        ax[0][i].set_xticklabels(frame.loc[class_key[i]].name, rotation=90, fontdict={'horizontalalignment': 'center', 'size':12})
        ax[0][i].set_ylim(0, 1)
        ax[0][i].set_xlim(-0.3,index[-1]+1)
        ax[1][i].set_xticks(index)
        ax[1][i].set_xticklabels(frame.loc[class_key[i]].name, rotation=90, fontdict={'horizontalalignment': 'center', 'size':12})
        ax[1][i].set_ylim(0, 1)
        ax[1][i].set_xlim(-0.3,index[-1]+1)   
    handles, labels = ax[0][0].get_legend_handles_labels()
    fig.legend(handles,labels,loc=7,prop={'size': 8})
    fig.tight_layout()
    fig.subplots_adjust(right=0.94)
    fig.savefig(name)

def compare_secondary_structures(model_dicts:list,name:str,model_labels=list):
    if len(model_dicts)>8:
        model_dicts=model_dicts[0:9]
    colors=sns.color_palette()
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    keys=['alpha','beta','loops','random']
    index=np.array([0,1,2,3])
    for j,model in enumerate(model_dicts):
        #show accuracy
        value=[model[k] for k in keys]
        ax[0].bar(x=index+j*0.1, height=value, width=0.1, align='center', color=colors[j],label=model_labels[j])
        #show top three accuracy
        if 'alpha_three' in model:
            value=[model[k+'_three'] for k in keys]
            ax[0].scatter(x=index+j*0.1, y=value,marker="_", s=50, color=colors[j])
            ax[0].vlines(x=index+j*0.1, ymin=0, ymax=value, color=colors[j], linewidth=2)
        #show recall
        value=[model[k+'_recall'] for k in keys]
        ax[1].bar(x=index+j*0.1, height=value, width=0.1, align='center', color=colors[j])
# Title, Label, Ticks and Ylim
        fig.suptitle('Secondary structure', fontdict={'size':22})
        ax[0].set_ylabel('Accuracy')
        ax[0].set_xticks(index)
        ax[0].set_xticklabels(['Helices','Sheets','Structured loops','Random'], rotation=90, fontdict={'horizontalalignment': 'center', 'size':12})
        ax[0].set_ylim(0, 1)
        ax[0].set_xlim(-0.3,index[-1]+1)

        ax[1].set_ylabel('Recall')
        ax[1].set_xticks(index)
        ax[1].set_xticklabels(['Helices','Sheets','Structured loops','Random'], rotation=90, fontdict={'horizontalalignment': 'center', 'size':12})
        ax[1].set_ylim(0, 1)
        ax[1].set_xlim(-0.3,index[-1]+1)
    plt.tight_layout()
    fig.legend(prop={'size': 8})
    fig.savefig(name)

def plot_resolution(df,predictions,name):
    colors=sns.color_palette()
    #combine class 4 and 6 to simplify the graph
    colors[6]=colors[4]
    class_color=[colors[x] for x in df['class'].values]
    accuracy,recall=get_cath.score_each(df,predictions)
    resolution=get_cath.get_resolution(df)
    corr=pd.DataFrame({0:resolution,1:recall,2:accuracy}).corr().to_numpy()
    fig, ax=plt.subplots(1,2,figsize=[10,5])
    ax[0].scatter(accuracy,resolution,color=class_color)
    ax[0].set_ylabel('Resolution, A')
    ax[0].set_xlabel('Accuracy')
    ax[0].set_title(f"Pearson correlation: {corr[0][2]:.3f}")
    ax[1].scatter(recall,resolution,color=class_color)
    ax[1].set_title(f"Pearson correlation: {corr[0][1]:.3f}")
    ax[1].set_xlabel('Recall')
    patches=[mpatches.Patch(color=colors[x], label=config.classes[x]) for x in config.classes]
    fig.legend(loc=1,handles=patches,prop={'size': 9})
    fig.tight_layout()
    fig.subplots_adjust(right=0.87)
    fig.savefig(name)


