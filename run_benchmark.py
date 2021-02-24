from benchmark import visualization
from benchmark import get_cath
from pathlib import Path
import click
import os

@click.command()
@click.option("--dataset", help="Path to .txt file with dataset list(PDB+chain, e.g., 1a2bA).")
@click.option("--path_to_pdb", help="Path to the directory with PDB files.")
@click.option("--path_to_assemblies", help="Path to the directory with biological assemblies.")
@click.option("--path_to_models", help="Path to the directory with .csv prediction files.")
@click.option("--path_to_dataset_map", help="Path to the .txt file with prediction labels.")
@click.option("--path_to_evoef", default=False, help="Path to the directory with EvoEF2 predictions. If supplied, EvoEF2 will be included in comparison.")
@click.option("--include", default=False, help="Path to .txt file with a list of models to be included in comparison. If not provided, 8 models with the best accuracy are compared.")
@click.option("--pdbs", default=False, help="Path to .txt file with a list of models and PDB codes for structures to be visualized. If not provided, no structures will be visualized.")
@click.option("--show_structures", default=False, help="Path to .txt file with a list of pdb structures for each model. Shows accuracy and entropy on each structure.")
@click.option("--by_fragment", default=False, help="Set to True if metrics should be calculated on a full chain, otherwise only CATH fragments are considered.")
@click.option("--ignore_uncommon", default=False, help="Set to True if your model ignores uncommon acids")
@click.option("--torsions", default=False, help="Set to True if you want to produce predicted and true Ramachandran plots for each model.")


def compare_models(dataset:str,path_to_pdb:str,path_to_assemblies:str,path_to_models:str,path_to_dataset_map:str,path_to_evoef:str,include:str,pdbs:str,ignore_uncommon:bool,by_fragment:bool,show_structures:str,torsions:bool):
    if include:
        with open(include) as file:
            models_to_include=[x.strip('\n') for x in file.readlines()]

    if pdbs:
        pdb_dict={}
        with open(pdbs) as file:
            for line in file.readlines():
                split_line=line.split()
                pdb_dict[split_line[0]]=[x.strip('\n') for x in split_line[1:]]
    df = get_cath.read_data('cath-domain-description-file.txt')
    filtered_df = get_cath.filter_with_user_list(df,dataset)
    df_with_sequence=get_cath.append_sequence(filtered_df,Path(path_to_assemblies),Path(path_to_pdb))
    #get_cath.compare(df_with_sequence,path_to_models,path_to_dataset_map,path_to_pdb,path_to_evoef,include,ignore_uncommon,by_fragment)

    accuracy = []
    list_of_models = {
        name: get_cath.load_prediction_matrix(df_with_sequence,path_to_dataset_map, Path(path_to_models) / name)
        for name in os.listdir(path_to_models)
        if name.split(".")[-1] == "csv"
    }
    for model in list_of_models:
        #make pdb visualization
        if pdbs:
            if model in pdb_dict:
                for protein in pdb_dict[model]:
                    visualization.show_accuracy(df_with_sequence,protein,list_of_models[model],Path(path_to_models)/f"{model}_{protein}.pdb",Path(path_to_pdb),ignore_uncommon,False)

        # make summary
        visualization.make_model_summary(df_with_sequence,list_of_models[model],str(Path(path_to_models)/model),Path(path_to_pdb))
        # get overall accuracy
        accuracy.append(
            [
                get_cath.score(
                    df_with_sequence,
                    list_of_models[model],
                    ignore_uncommon=ignore_uncommon,
                    by_fragment=by_fragment,
                )[0][0],
                model,
            ]
        )
        #make Ramachandran plots
        if torsions:
            sequence, prediction,_, angle=get_cath.format_angle_sequence(df_with_sequence,list_of_models[model],Path(path_to_assemblies),ignore_uncommon=ignore_uncommon,by_fragment=by_fragment)
            visualization.ramachandran_plot(sequence, list(get_cath.most_likely_sequence(prediction)), angle, str(Path(path_to_models)/model))
    # load evoef predictions and make evoef summary
    if path_to_evoef:
        evo_ef2 = get_cath.load_prediction_sequence(df_with_sequence, path_to_evoef)
        visualization.make_model_summary(
            df_with_sequence,
            evo_ef2,
            str(Path(path_to_models) / "EvoEF2"),
            Path(path_to_pdb),
            ignore_uncommon=True,
            score_sequence=True,
        )
        if torsions:
            sequence, prediction,_, angle=get_cath.format_angle_sequence(df_with_sequence,evo_ef2,Path(path_to_assemblies),ignore_uncommon=True,by_fragment=by_fragment,score_sequence=True)
            visualization.ramachandran_plot(sequence, prediction, angle, str(Path(path_to_models)/model))

    accuracy = sorted(accuracy)
    if path_to_evoef:
        # pick 7 best models and evoef
        filtered_models = [list_of_models[model[1]] for model in accuracy[-7:]]
        filtered_labels = [model[1] for model in accuracy[-7:]]
        # add evoEF2 data
        filtered_models.append(evo_ef2)
        filtered_labels.append("EvoEF2")
    else:
        filtered_models = [list_of_models[model[1]] for model in accuracy[-8:]]
        filtered_labels = [model[1] for model in accuracy[-8:]]
    # include specified models
    if include:
        if len(models_to_include) <= 8:
            for index, model_name in enumerate(models_to_include):
                if model_name not in filtered_labels:
                    filtered_models[index] = list_of_models[model_name]
                    filtered_labels[index] = model_name
        else:
            raise ValueError(
                "Too many models are give to plot, select no more than 8 models."
            )

    visualization.compare_model_accuracy(
        df_with_sequence,
        filtered_models,
        filtered_labels,
        Path(path_to_models),
        ignore_uncommon,
        by_fragment,
    )
if __name__ == '__main__':
    compare_models()