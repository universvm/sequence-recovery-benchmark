"""Functions for visualizing metrics and comparing different models"""

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
from sklearn import metrics
import matplotlib.backends.backend_pdf
from scipy.stats import entropy
from typing import List
from benchmark import version

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
        Ampal structure with modified B-factor and occupancy values.

    Notes
    -----
    Leo's code.
    Same as _annotate_ampalobj_with_data_tag from TIMED but can deal with missing unnatural amino acids for compatibility with EvoEF2."""

    assert len(tags) == len(
        data_to_annotate
    ), "The number of tags to annotate and the type of data to annotate have different lengths."

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
    for i, tag in enumerate(tags):

        # Check if chain is Polypeptide (it might be DNA for example...)
        if isinstance(ampal_structure, ampal.Polypeptide):
            if len(ampal_structure) != len(data_to_annotate[i]):
                # EvoEF2 predictions drop uncommon amino acids
                if len(ampal_structure) - ampal_structure.sequence.count("X") == len(
                    data_to_annotate[i]
                ):
                    for residue in ampal_structure:
                        counter = 0
                        if ampal.amino_acids.get_aa_letter(residue) == "X":
                            continue
                        else:
                            for atom in residue:
                                atom.tags[tag] = data_to_annotate[i][counter]
                                counter += 1
                else:
                    print("Length is not equal")
                    return
            for residue, data_val in zip(ampal_structure, data_to_annotate[i]):
                for atom in residue:
                    atom.tags[tag] = data_val

    return ampal_structure


def show_accuracy(
    df: pd.DataFrame,
    pdb: str,
    predictions: dict,
    output: Path,
    path_to_pdbs: Path,
    ignore_uncommon: bool,
) -> None:
    """
    Parameters
    ----------
    df: pd.DataFrame
        CATH dataframe.
    pdb: str
      PDB code to visualize, format: pdb+CHAIN.
    predictions: dict
        Dictionary with predicted sequences, key is PDB+chain.
    name: str
        Location of the .pdf file, also title of the plot.
    output: Path
        Path to output directory.
    path_to_pdbs: Path
        Path to the directory with PDB files.
    ignore_uncommon=True
        If True, ignores uncommon residues in accuracy calculations.
    score_sequence=False
        True if dictionary contains sequences, False if probability matrices(matrix shape n,20)."""
    accuracy = []
    pdb_df = df[df.PDB == pdb]
    sequence, prediction, _, _, _ = get_cath.format_sequence(
        pdb_df, predictions, False, ignore_uncommon,
    )
    
    entropy_arr = entropy(prediction, base=2, axis=1)
    prediction = list(get_cath.most_likely_sequence(prediction))
    for resa, resb in zip(sequence, prediction):
        """correct predictions are given constant score so they stand out in the figure.
        e.g., spectrum q, blue_white_red, maximum=6,minimum=-6 gives nice plots. Bright red shows correct predictions
        Red shades indicate substitutions with positive score, white=0, blue shades show substiutions with negative score.
        cartoon putty shows nice entropy visualization."""

        if resa == resb:
            accuracy.append(6)
        # incorrect predictions are coloured by blossum62 score.
        else:
            accuracy.append(get_cath.lookup_blosum62(resa, resb))
    path_to_protein = path_to_pdbs / pdb[1:3] / f"pdb{pdb}.ent.gz"
    with gzip.open(path_to_protein, "rb") as protein:
        assembly = ampal.load_pdb(protein.read().decode(), path=False)

    # Deals with structures from NMR as ampal returns Container of Assemblies
    if isinstance(assembly, ampal.AmpalContainer):
        warnings.warn(f"Selecting the first state from the NMR structure {assembly.id}")
        assembly = assembly[0]
    # select correct chain
    assembly = assembly[pdb_df.chain.values[0]]
    
    curr_annotated_structure = _annotate_ampalobj_with_data_tag(
        assembly, [accuracy, entropy_arr], tags=["occupancy","bfactor"]
    )
    with open(output, "w") as f:
        f.write(curr_annotated_structure.pdb)


def ramachandran_plot(
    sequence: List[chr], prediction: List[chr], torsions: List[List[float]], name: str
) -> None:
    """Plots predicted and true Ramachandran plots for each amino acid. All plots are normalized by true residue count. Takes at least a minute to plot these, so don't plot if not neccessary.
    Parameters
    ----------
    sequence: List[chr]
        List with correctly formated (get_cath.format_format_angle_sequence()) sequence.
    prediction: List[chr]
        List with correctly formated predictions. Amino acid sequence, not arrays.
    torsions: List[List[float]]
        List wit correctly formated torsion angles.
    name: str
        Name and location of the figure."""

    fig, ax = plt.subplots(20, 3, figsize=(15, 100))
    plt.figtext(0.1, 0.99,s='Version: '+version.__version__,figure=fig,fontdict={"size": 12})
    # get angles for each amino acids
    for k, amino_acid in enumerate(config.acids):
        predicted_angles = [
            x for x, residue in zip(torsions, prediction) if residue == amino_acid
        ]
        predicted_psi = [
            x[2] for x in predicted_angles if (x[2] != None) & (x[1] != None)
        ]
        predicted_phi = [
            x[1] for x in predicted_angles if (x[1] != None) & (x[2] != None)
        ]

        true_angles = [
            x for x, residue in zip(torsions, list(sequence)) if residue == amino_acid
        ]
        true_psi = [x[2] for x in true_angles if (x[2] != None) & (x[1] != None)]
        true_phi = [x[1] for x in true_angles if (x[1] != None) & (x[2] != None)]

        # make a histogram and normalize by residue count
        array, xedges, yedges = [
            x
            for x in np.histogram2d(
                predicted_psi, predicted_phi, bins=50, range=[[-180, 180], [-180, 180]]
            )
        ]
        array = array / len(true_psi)
        true_array, xedges, yedges = [
            x
            for x in np.histogram2d(
                true_psi, true_phi, bins=50, range=[[-180, 180], [-180, 180]]
            )
        ]
        true_array = true_array / len(true_psi)
        difference = true_array - array
        # get minimum and maximum counts for true and predicted sequences, use this to keep color maping in both plots identical. Easier to see overprediction.
        minimum = np.amin([array, true_array])
        maximum = np.amax([array, true_array])
        # change 0 counts to NaN to show white space.
        # make Ramachandran plot for predictions.
        for i, rows in enumerate(array):
            for j, cols in enumerate(rows):
                if cols == 0.0:
                    array[i][j] = np.NaN

        im = ax[k][0].imshow(
            array,
            interpolation="none",
            norm=None,
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap="viridis",
            vmax=maximum,
            vmin=minimum,
        )
        fig.colorbar(im, ax=ax[k][0], fraction=0.046)
        ax[k][0].set_xlim(-180, 180)
        ax[k][0].set_ylim(-180, 180)
        ax[k][0].set_xticks(np.arange(-180, 220, 40))
        ax[k][0].set_yticks(np.arange(-180, 220, 40))
        ax[k][0].set_ylabel("Psi")
        ax[k][0].set_xlabel("Phi")
        ax[k][0].set_title(f"Predicted {amino_acid}")

        # Make Ramachandran plot for true sequence.
        for i, rows in enumerate(true_array):
            for j, cols in enumerate(rows):
                if cols == 0.0:
                    true_array[i][j] = np.NaN
        im = ax[k][1].imshow(
            true_array,
            interpolation="none",
            norm=None,
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap="viridis",
            vmax=maximum,
            vmin=minimum,
        )
        fig.colorbar(im, ax=ax[k][1], fraction=0.046)
        ax[k][1].set_xlim(-180, 180)
        ax[k][1].set_ylim(-180, 180)
        ax[k][1].set_xticks(np.arange(-180, 220, 40))
        ax[k][1].set_yticks(np.arange(-180, 220, 40))
        ax[k][1].set_ylabel("Psi")
        ax[k][1].set_xlabel("Phi")
        ax[k][1].set_title(f"True {amino_acid}")

        # Make difference plots.
        for i, rows in enumerate(difference):
            for j, cols in enumerate(rows):
                if cols == 0.0:
                    difference[i][j] = np.NaN

        im = ax[k][2].imshow(
            difference,
            interpolation="none",
            norm=None,
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap="viridis",
        )
        fig.colorbar(im, ax=ax[k][2], fraction=0.046)
        ax[k][2].set_xlim(-180, 180)
        ax[k][2].set_ylim(-180, 180)
        ax[k][2].set_xticks(np.arange(-180, 220, 40))
        ax[k][2].set_yticks(np.arange(-180, 220, 40))
        ax[k][2].set_ylabel("Psi")
        ax[k][2].set_xlabel("Phi")
        ax[k][2].set_title(f"True-Predicted {amino_acid}")

    plt.tight_layout()
    plt.savefig(name + "_Ramachandran_plot.pdf")
    plt.close()


def append_zero_residues(arr: np.array) -> np.array:
    """Sets missing residue count to 0. Needed for per residue metrics plot.
    Parameters
    ----------
    arr:np.array
        Array returned by np.unique() with residues and their counts.
    Returns
    -------
    np.array with added mising residues and 0 counts."""
    if len(arr[0]) != 20:
        temp_dict = {res_code: res_count for res_code, res_count in zip(arr[0], arr[1])}
        for residue in config.acids:
            if residue not in temp_dict:
                temp_dict[residue] = 0
        arr = [[], []]
        arr[1] = [x[1] for x in sorted(temp_dict.items())]
        arr[0] = [x[0] for x in sorted(temp_dict.items())]
    return arr


def make_model_summary(
    df: pd.DataFrame,
    predictions: dict,
    name: str,
    path_to_pdb: Path,
    ignore_uncommon: bool = False,
) -> None:
    """
    Makes a .pdf report whith model metrics.
    Includes prediction bias, accuracy and macro recall for each secondary structure, accuracy and recall correlation with protein resolution, confusion matrices and accuracy, recall and f1 score for each resiude.

    Parameters
    ----------
    df: pd.DataFrame
        CATH dataframe.
    predictions: dict
        Dictionary with predicted sequences, key is PDB+chain.
    name: str
        Location of the .pdf file, also title of the plot.
    path_to_pdb: Path
        Path to the directory with PDB files.
    ignore_uncommon=True
        If True, ignores uncommon residues in accuracy calculations.
    """
    
    fig, ax = plt.subplots(ncols=5, nrows=5, figsize=(30, 40))
    #print version
    plt.figtext(0.1, 0.99,s='Version: '+version.__version__,figure=fig,fontdict={"size": 12})
    # show residue distribution and confusion matrix
    (
        sequence,
        prediction,
        _,
        true_secondary,
        prediction_secondary,
    ) = get_cath.format_sequence(
        df,
        predictions,
        ignore_uncommon=ignore_uncommon,
        by_fragment=False,
    )
    # get info about each residue
    by_residue_frame = get_cath.get_by_residue_metrics(
        sequence, prediction
    )
    # convert probability array into list of characters.
    prediction = list(get_cath.most_likely_sequence(prediction))
    prediction_secondary = [
        list(get_cath.most_likely_sequence(ss_seq))
        for ss_seq in prediction_secondary
    ]

    seq = append_zero_residues(np.unique(sequence, return_counts=True))

    pred = append_zero_residues(np.unique(prediction, return_counts=True))
    index = np.arange(len(seq[0]))
    # calculate prediction bias
    residue_bias = pred[1] / sum(pred[1]) - seq[1] / sum(seq[1])
    ax[3][4].bar(x=index, height=residue_bias, width=0.8, align="center")
    ax[3][4].set_ylabel("Prediction bias")
    ax[3][4].set_xlabel("Amino acids")
    for e, dif in enumerate(residue_bias):
        if dif < 0:
            y_coord = 0
        else:
            y_coord = dif
        ax[3][4].text(
            index[e],
            y_coord + 0.03,
            f"{dif:.3f}",
            ha="center",
            va="bottom",
            rotation="vertical",
        )

    ax[3][4].set_xticks(index)
    ax[3][4].set_xticklabels(
        pred[0], fontdict={"horizontalalignment": "center", "size": 12}
    )
    ax[3][4].set_ylabel("Prediction bias")
    ax[3][4].set_xlabel("Amino acids")
    ax[3][4].set_title("All structures")
    ax[3][4].set_ylim(top=1.0)

    cm = metrics.confusion_matrix(sequence, prediction, labels=seq[0])
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    im = ax[4][4].imshow(cm, vmin=0, vmax=1)
    ax[4][4].set_xlabel("Predicted")
    ax[4][4].set_xticks(range(20))
    ax[4][4].set_xticklabels(config.acids)
    ax[4][4].set_ylabel("True")
    ax[4][4].set_yticks(range(20))
    ax[4][4].set_yticklabels(config.acids)
    # Plot Color Bar:
    fig.colorbar(im, ax=ax[4][4], fraction=0.046)

    # plot prediction bias
    ss_names = ["Helices", "Sheets", "Structured loops", "Random"]
    for i, ss in enumerate(ss_names):
        seq = append_zero_residues(np.unique(true_secondary[i], return_counts=True))
        pred = append_zero_residues(
            np.unique(prediction_secondary[i], return_counts=True)
        )
        residue_bias = pred[1] / sum(pred[1]) - seq[1] / sum(seq[1])
        ax[3][i].bar(x=index, height=residue_bias, width=0.8, align="center")
        ax[3][i].set_xticks(index)
        ax[3][i].set_xticklabels(
            pred[0], fontdict={"horizontalalignment": "center", "size": 12}
        )
        ax[3][i].set_ylabel("Prediction bias")
        ax[3][i].set_xlabel("Amino acids")
        ax[3][i].set_title(ss)
        ax[3][i].set_ylim(top=1.0)
        for e, dif in enumerate(residue_bias):
            if dif < 0:
                y_coord = 0
            else:
                y_coord = dif
            ax[3][i].text(
                index[e],
                y_coord + 0.03,
                f"{dif:.3f}",
                ha="center",
                va="bottom",
                rotation="vertical",
            )
        #plot confusion matrix
        cm = metrics.confusion_matrix(
            true_secondary[i], prediction_secondary[i], labels=seq[0]
        )
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        im = ax[4][i].imshow(cm, vmin=0, vmax=1)
        ax[4][i].set_xlabel("Predicted")
        ax[4][i].set_xticks(range(20))
        ax[4][i].set_xticklabels(config.acids)
        ax[4][i].set_ylabel("True")
        ax[4][i].set_yticks(range(20))
        ax[4][i].set_yticklabels(config.acids)
        # Plot Color Bar:
        fig.colorbar(im, ax=ax[4][i], fraction=0.046)

    # show accuracy,recall,similarity, precision and top3
    index = np.array([0, 1, 2, 3, 4])
    
    accuracy, top_three, similarity, recall, precision = get_cath.score(
        df, predictions, False, ignore_uncommon,
    )
    # show accuracy
    ax[0][0].bar(x=index, height=accuracy, width=0.8, align="center")

    # show top three accuracy
    ax[0][0].scatter(x=index, y=top_three, marker="_", s=50, color="blue")
    ax[0][0].vlines(x=index, ymin=0, ymax=top_three, linewidth=2)
    # show recall
    ax[0][1].bar(x=index, height=recall, width=0.8, align="center")
    ax[0][3].bar(x=index, height=precision, width=0.8, align="center")
    ax[0][4].bar(x=index, height=similarity, width=0.8, align="center")
    # add values to the plot
    for e, value in enumerate(accuracy):
        ax[0][0].text(
            index[e],
            value + 0.3,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            rotation="vertical",
        )
    for e, value in enumerate(recall):
        ax[0][1].text(
            index[e],
            value * 1.2,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            rotation="vertical",
        )
    for e, value in enumerate(precision):
        ax[0][3].text(
            index[e],
            value * 1.2,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            rotation="vertical",
        )
    for e, value in enumerate(similarity):
        ax[0][4].text(
            index[e],
            value * 1.2,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            rotation="vertical",
        )
    # show difference

    difference = np.array(accuracy) - np.array(recall)
    minimum = np.amin(difference)
    maximum = np.amax(difference)
    ax[0][2].bar(x=index, height=difference, width=0.8, align="center")
    for e, dif in enumerate(difference):
        if dif < 0:
            y_coord = 0
        else:
            y_coord = dif
        ax[0][2].text(
            index[e],
            y_coord + 0.01,
            f"{dif:.3f}",
            ha="center",
            va="bottom",
            rotation="vertical",
        )
    # Title, label, ticks and limits
    ax[0][0].set_ylabel("Accuracy")
    ax[0][0].set_xticks(index)
    ax[0][0].set_xticklabels(
        ["All structures", "Helices", "Sheets", "Structured loops", "Random"],
        rotation=90,
        fontdict={"horizontalalignment": "center", "size": 12},
    )
    ax[0][0].set_ylim(0, 1)
    ax[0][0].set_xlim(-0.7, index[-1] + 1)

    ax[0][1].set_ylabel("Average recall")
    ax[0][1].set_xticks(index)
    ax[0][1].set_xticklabels(
        ["All structures", "Helices", "Sheets", "Structured loops", "Random"],
        rotation=90,
        fontdict={"horizontalalignment": "center", "size": 12},
    )
    ax[0][1].set_ylim(0, 1)
    ax[0][1].set_xlim(-0.7, index[-1] + 1)

    ax[0][2].set_ylabel("Accuracy-Average recall")
    ax[0][2].set_xticks(index)
    ax[0][2].set_xticklabels(
        ["All structures", "Helices", "Sheets", "Structured loops", "Random"],
        rotation=90,
        fontdict={"horizontalalignment": "center", "size": 12},
    )
    ax[0][2].set_xlim(-0.7, index[-1] + 1)
    ax[0][2].axhline(0, -0.3, index[-1] + 1, color="k", lw=1)
    ax[0][2].set_ylim(minimum * 1.2, maximum * 1.2)

    ax[0][3].set_ylabel("Average precision")
    ax[0][3].set_xticks(index)
    ax[0][3].set_xticklabels(
        ["All structures", "Helices", "Sheets", "Structured loops", "Random"],
        rotation=90,
        fontdict={"horizontalalignment": "center", "size": 12},
    )
    ax[0][3].set_ylim(0, 1)
    ax[0][3].set_xlim(-0.7, index[-1] + 1)

    ax[0][4].set_ylabel("Similarity")
    ax[0][4].set_xticks(index)
    ax[0][4].set_xticklabels(
        ["All structures", "Helices", "Sheets", "Structured loops", "Random"],
        rotation=90,
        fontdict={"horizontalalignment": "center", "size": 12},
    )
    ax[0][4].set_ylim(0, 1)
    ax[0][4].set_xlim(-0.7, index[-1] + 1)

    colors = sns.color_palette("viridis", 4)
    # combine classes 4 and 6 to simplify the graph
    colors = {1: colors[0], 2: colors[1], 3: colors[2], 4: colors[3], 6: colors[3]}
    class_color = [colors[x] for x in df["class"].values]
    # show accuracy and macro recall resolution distribution
    accuracy, recall = get_cath.score_each(
        df,
        predictions,
        ignore_uncommon=ignore_uncommon,
        by_fragment=True,
    )
    resolution = get_cath.get_resolution(df, path_to_pdb)
    # calculate Pearson correlation between accuracy/recall and resolution.
    corr = pd.DataFrame({0: resolution, 1: recall, 2: accuracy}).corr().to_numpy()
    ax[1][3].scatter(resolution, accuracy, color=class_color, alpha=0.7)
    # Title, label, ticks and limits
    ax[1][3].set_xlabel("Resolution, A")
    ax[1][3].set_ylabel("Accuracy")
    ax[1][3].set_title(f"Pearson correlation: {corr[0][2]:.3f}")
    ax[1][4].scatter(resolution, recall, color=class_color, alpha=0.7)
    ax[1][4].set_title(f"Pearson correlation: {corr[0][1]:.3f}")
    ax[1][4].set_ylabel("Average recall")
    ax[1][4].set_xlabel("Resolution, A")
    # make a legend
    patches = [
        mpatches.Patch(color=colors[x], label=config.classes[x]) for x in config.classes
    ]
    ax[1][4].legend(loc=1, handles=patches, prop={"size": 9})
    ax[1][3].legend(loc=1, handles=patches, prop={"size": 9})
    # show per residue metrics about the model
    gs = ax[0, 0].get_gridspec()
    # show per residue entropy
    ax[2][0].bar(by_residue_frame.index, by_residue_frame.entropy)
    ax[2][0].set_ylabel("Entropy")
    ax[2][0].set_xlabel("Amino acids")

    # make one big subplot
    for a in ax[2, 1:]:
        a.remove()
    ax_right = fig.add_subplot(gs[2, 1:])
    index = np.arange(len(by_residue_frame.index))
    # show recall,precision and f1
    for i, metric in enumerate(["recall", "precision", "f1"]):
        ax_right.bar(
            index + i * 0.3, height=by_residue_frame[metric], width=0.3, label=metric
        )
        # add values to the plot
        for j, value in enumerate(by_residue_frame[metric]):
            ax_right.text(
                index[j] + i * 0.3,
                value + 0.05,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                rotation="vertical",
            )
    ax_right.legend()
    ax_right.set_xticks(index + 0.3)
    ax_right.set_xticklabels(
        by_residue_frame.index, fontdict={"horizontalalignment": "center", "size": 12}
    )
    ax_right.set_xlim(index[0] - 0.3, index[-1] + 1)
    ax_right.set_ylim(0, 1)

    #show auc values
    ax[1][0].bar(by_residue_frame.index, by_residue_frame.auc)
    ax[1][0].set_ylabel("AUC")
    ax[1][0].set_xlabel("Amino acids")
    #Remove empty subplots.
    ax[1][1].remove()
    ax[1][2].remove()



    plt.suptitle(name, fontsize="xx-large")
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(name + ".pdf")
    plt.close()


def compare_model_accuracy(
    df: pd.DataFrame,
    model_scores: List[dict],
    model_labels: List[str],
    location: Path,
    ignore_uncommon: List[bool],
) -> None:
    """
    Compares all the models in model_scores.
    .pdf report contains accuracy, macro average and similarity scores for each CATH architecture and secondary structure type.

    Parameters
    ----------

    df: pd.DataFrame
        CATH dataframe.
    model_scores: List[dict]
        List with dictionary with predicted sequences.
    model_labels: List[str]
        List with model names corresponding to dictionaries in model_scores.
    location:Path
        Location where to store the .pdf file.
    ignore_uncommon=List[bool]
        If True, ignores uncommon residues in accuracy calculations. Required for EvoEF2."""

    models = []
    for model, ignore in zip(model_scores, ignore_uncommon):
        models.append(
            get_cath.score_by_architecture(
                df,
                model,
                ignore_uncommon=ignore,
                by_fragment=True,
            )
        )
    # Plot CATH architectures
    minimum = 0
    maximum = 0
    colors = sns.color_palette()
    # combine classes 4 and 6 to make plots nicer. Works with any number of CATH classes.
    class_key = [x[0] for x in models[0].index]
    class_key = list(dict.fromkeys(class_key))
    if 4 in class_key and 6 in class_key:
        class_key = [x for x in class_key if x != 4 and x != 6]
        class_key.append([4, 6])
    # calculate subplot ratios so that classes with more architectures have more space.
    ratios = [models[0].loc[class_key[i]].shape[0] for i in range(len(class_key))]
    fig, ax = plt.subplots(
        5,
        len(class_key),
        figsize=(12 * len(class_key), 20),
        gridspec_kw={"width_ratios": ratios},
        squeeze=False,
    )
    plt.figtext(0.1, 0.99,s='Version: '+version.__version__,figure=fig,fontdict={"size": 12})

    for i in range(len(class_key)):
        index = np.arange(0, models[0].loc[class_key[i]].shape[0])
        for j, frame in enumerate(models):
            value_accuracy = frame.loc[class_key[i]].accuracy.values
            value_recall = frame.loc[class_key[i]].recall.values
            value_similarity = frame.loc[class_key[i]].similarity.values
            # show accuracy
            ax[0][i].bar(
                x=index + j * 0.1,
                height=value_accuracy,
                width=0.1,
                align="center",
                color=colors[j],
                label=model_labels[j],
            )
            for e, accuracy in enumerate(value_accuracy):
                ax[0][i].text(
                    index[e] + j * 0.1,
                    accuracy + 0.3,
                    f"{accuracy:.3f}",
                    ha="center",
                    va="bottom",
                    rotation="vertical",
                    fontdict={"size": 7},
                )
            # show recall
            ax[1][i].bar(
                x=index + j * 0.1,
                height=value_recall,
                width=0.1,
                align="center",
                color=colors[j],
            )
            for e, recall in enumerate(value_recall):
                ax[1][i].text(
                    index[e] + j * 0.1,
                    recall * 1.2,
                    f"{recall:.3f}",
                    ha="center",
                    va="bottom",
                    rotation="vertical",
                    fontdict={"size": 7},
                )

            # show top3 accuracy if it exists
            if "top3_accuracy" in frame:
                value_top_three = frame.loc[class_key[i]].top3_accuracy.values
                ax[0][i].scatter(
                    x=index + j * 0.1,
                    y=value_top_three,
                    marker="_",
                    s=50,
                    color=colors[j],
                )
                ax[0][i].vlines(
                    x=index + j * 0.1,
                    ymin=0,
                    ymax=value_top_three,
                    color=colors[j],
                    linewidth=2,
                )
            # show similarity scores
            ax[2][i].bar(
                x=index + j * 0.1,
                height=value_similarity,
                width=0.1,
                align="center",
                color=colors[j],
            )
            for e, similarity in enumerate(value_similarity):
                ax[2][i].text(
                    index[e] + j * 0.1,
                    similarity * 1.2,
                    f"{similarity:.3f}",
                    ha="center",
                    va="bottom",
                    rotation="vertical",
                    fontdict={"size": 7},
                )
            # show accuracy-macro recall
            difference = value_accuracy - value_recall
            if np.amin(difference) < minimum:
                minimum = np.amin(difference)
            if np.amax(difference) > maximum:
                maximum = np.amax(difference)
            ax[3][i].bar(
                x=index + j * 0.1,
                height=difference,
                width=0.1,
                align="center",
                color=colors[j],
            )
            for e, dif in enumerate(difference):
                if dif < 0:
                    y_coord = 0
                else:
                    y_coord = dif
                ax[3][i].text(
                    index[e] + j * 0.1,
                    y_coord + 0.01,
                    f"{dif:.3f}",
                    ha="center",
                    va="bottom",
                    rotation="vertical",
                    fontdict={"size": 7},
                )

        # Title, Label, Ticks and Ylim
        ax[0][i].set_title(config.classes[i + 1], fontdict={"size": 22})
        ax[1][i].set_title(config.classes[i + 1], fontdict={"size": 22})
        ax[2][i].set_title(config.classes[i + 1], fontdict={"size": 22})
        ax[3][i].set_title(config.classes[i + 1], fontdict={"size": 22})
        ax[0][i].set_ylabel("Accuracy")
        ax[1][i].set_ylabel("AverageRecall")
        ax[2][i].set_ylabel("Similarity")
        ax[3][i].set_ylabel("Accuracy-AverageRecall")
        ax[0][i].set_xticks(index)
        ax[0][i].set_xticklabels(
            frame.loc[class_key[i]].name,
            rotation=90,
            fontdict={"horizontalalignment": "center", "size": 12},
        )
        ax[0][i].set_ylim(0, 1)
        ax[0][i].set_xlim(-0.3, index[-1] + 1)
        ax[1][i].set_xticks(index)
        ax[1][i].set_xticklabels(
            frame.loc[class_key[i]].name,
            rotation=90,
            fontdict={"horizontalalignment": "center", "size": 12},
        )
        ax[1][i].set_ylim(0, 1)
        ax[1][i].set_xlim(-0.3, index[-1] + 1)
        ax[2][i].set_xticks(index)
        ax[2][i].set_xticklabels(
            frame.loc[class_key[i]].name,
            rotation=90,
            fontdict={"horizontalalignment": "center", "size": 12},
        )
        ax[2][i].set_ylim(0, 1)
        ax[2][i].set_xlim(-0.3, index[-1] + 1)
        ax[3][i].set_xticks(index)
        ax[3][i].set_xticklabels(
            frame.loc[class_key[i]].name,
            rotation=90,
            fontdict={"horizontalalignment": "center", "size": 12},
        )
        ax[3][i].hlines(0, -0.3, index[-1] + 1, colors="k", lw=1)
        ax[3][i].set_xlim(-0.3, index[-1] + 1)
    # Make yaxis in difference plots equal to get a nice graph.
    for x in range(len(ax[3])):
        ax[3][x].set_ylim(minimum * 1.2, maximum * 1.2)
    handles, labels = ax[0][0].get_legend_handles_labels()
    ax[4][0].legend(handles, labels, loc=1, prop={"size": 12})
    ax[4][0].set_axis_off()
    for x in range(1, len(class_key)):
        ax[4][x].remove()
    fig.tight_layout()

    # Plot secondary structures
    maximum = 0
    minimum = 0
    fig_secondary, ax_secondary = plt.subplots(2, 2, figsize=(12 * len(class_key), 10))
    index = np.array([0, 1, 2, 3, 4])
    for j, model in enumerate(model_scores):
        accuracy, top_three, similarity, recall, precision = get_cath.score(
            df, model, False, ignore_uncommon[j],
        )
        # show accuracy
        ax_secondary[0][0].bar(
            x=index + j * 0.1,
            height=accuracy,
            width=0.1,
            align="center",
            color=colors[j],
            label=model_labels[j],
        )
        # show top three accuracy
        ax_secondary[0][0].scatter(
            x=index + j * 0.1, y=top_three, marker="_", s=50, color=colors[j]
        )
        ax_secondary[0][0].vlines(
            x=index + j * 0.1, ymin=0, ymax=top_three, color=colors[j], linewidth=2
        )
        # show recall
        ax_secondary[0][1].bar(
            x=index + j * 0.1,
            height=recall,
            width=0.1,
            align="center",
            color=colors[j],
            label=model_labels[j],
        )
        # show similarity score
        ax_secondary[1][1].bar(
            x=index + j * 0.1,
            height=similarity,
            width=0.1,
            align="center",
            color=colors[j],
            label=model_labels[j],
        )
        # add values to the plot
        for e, value in enumerate(accuracy):
            ax_secondary[0][0].text(
                index[e] + j * 0.1,
                value + 0.3,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                rotation="vertical",
                fontdict={"size": 6},
            )
        for e, value in enumerate(recall):
            ax_secondary[0][1].text(
                index[e] + j * 0.1,
                value * 1.2,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                rotation="vertical",
                fontdict={"size": 6},
            )
        for e, value in enumerate(similarity):
            ax_secondary[1][1].text(
                index[e] + j * 0.1,
                value * 1.2,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                rotation="vertical",
                fontdict={"size": 6},
            )
        # show difference
        difference = np.array(accuracy) - np.array(recall)
        if np.amin(difference) < minimum:
            minimum = np.amin(difference)
        if np.amax(difference) > maximum:
            maximum = np.amax(difference)
        ax_secondary[1][0].bar(
            x=index + j * 0.1,
            height=difference,
            width=0.1,
            align="center",
            color=colors[j],
        )
        for e, dif in enumerate(difference):
            if dif < 0:
                y_coord = 0
            else:
                y_coord = dif
            ax_secondary[1][0].text(
                e + j * 0.1,
                y_coord + 0.01,
                f"{dif:.3f}",
                ha="center",
                va="bottom",
                rotation="vertical",
                fontdict={"size": 6},
            )
        # Title, labels, ticks and limits
        fig_secondary.suptitle("Secondary structure", fontdict={"size": 22})
        ax_secondary[0][0].set_ylabel("Accuracy")
        ax_secondary[0][0].set_xticks([0, 1, 2, 3, 4])
        ax_secondary[0][0].set_xticklabels(
            ["All structures", "Helices", "Sheets", "Structured loops", "Random"],
            rotation=90,
            fontdict={"horizontalalignment": "center", "size": 12},
        )
        ax_secondary[0][0].set_ylim(0, 1)
        # leave some space from the sides to make it look nicer.
        ax_secondary[0][0].set_xlim(-0.3, 5)

        ax_secondary[0][1].set_ylabel("Average Recall")
        ax_secondary[0][1].set_xticks([0, 1, 2, 3, 4])
        ax_secondary[0][1].set_xticklabels(
            ["All structures", "Helices", "Sheets", "Structured loops", "Random"],
            rotation=90,
            fontdict={"horizontalalignment": "center", "size": 12},
        )
        ax_secondary[0][1].set_ylim(0, 1)
        ax_secondary[0][1].set_xlim(-0.3, 5)

        ax_secondary[1][1].set_ylabel("Similarity")
        ax_secondary[1][1].set_xticks([0, 1, 2, 3, 4])
        ax_secondary[1][1].set_xticklabels(
            ["All structures", "Helices", "Sheets", "Structured loops", "Random"],
            rotation=90,
            fontdict={"horizontalalignment": "center", "size": 12},
        )
        ax_secondary[1][1].set_ylim(0, 1)
        ax_secondary[1][1].set_xlim(-0.3, 5)

        ax_secondary[1][0].set_ylabel("Accuracy-Average Recall")
        ax_secondary[1][0].set_xticks([0, 1, 2, 3, 4])
        ax_secondary[1][0].set_xticklabels(
            ["All structures", "Helices", "Sheets", "Structured loops", "Random"],
            rotation=90,
            fontdict={"horizontalalignment": "center", "size": 12},
        )
        ax_secondary[1][0].set_xlim(-0.3, 5)
        ax_secondary[1][0].axhline(0, -0.3, index[-1] + 1, color="k", lw=1)
        # make y axis in difference plots equal to get nicer graphs.
        ax_secondary[1][0].set_ylim(minimum * 1.2, maximum * 1.2)
    fig_secondary.tight_layout()

    pdf = matplotlib.backends.backend_pdf.PdfPages(location / "Comparison_summary.pdf")
    pdf.savefig(fig)
    pdf.savefig(fig_secondary)
    pdf.close()
    plt.close()
    
