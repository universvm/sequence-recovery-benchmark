from benchmark import visualization
from benchmark import get_cath
from pathlib import Path
import click

'''
PATH_TO_PDB = Path("/home/shared/datasets/pdb/")
PATH_TO_ASSEMBLIES = Path("/home/shared/datasets/biounit/")
'''
'''Amino acid key
['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
'''
@click.command()
@click.option("--dataset", help="Path to .txt file with dataset list(PDB+chain, e.g., 1a2bA).")
@click.option("--path_to_pdb", help="Path to the directory with PDB files.")
@click.option("--path_to_assemblies", help="Path to the directory with biological assemblies.")
@click.option("--path_to_models", help="Path to the directory with .csv prediction files.")
@click.option("--path_to_dataset_map", help="Path to the .txt file with prediction labels.")
@click.option("--acid_key", help="Order of amino acids in the probability matrix.")
@click.option("--path_to_evoef", default=False, help="Path to the directory with EvoEF2 predictions. If supplied, EvoEF2 will be included in comparison.")
@click.option("--include", default=False, help="A list of models to be included in comparison. If not provided, 8 models with the best accuracy are compared.")
@click.option("--by_fragment", default=False, help="Set to True if metrics should be calculated on a full chain, otherwise only CATH fragments are considered.")
@click.option("--ignore_uncommon", default=False, help="Set to True if your model ignores uncommon acids")
def compare_models(dataset:str,path_to_pdb:str,path_to_assemblies:str,path_to_models:str,path_to_dataset_map:str,acid_key:str,path_to_evoef:str=False,include:list=False,ignore_uncommon:bool=False,by_fragment:bool=True):
    if len(acid_key)==20:
        acid_key=list(acid_key)
    else:
        raise ValueError('The key must contain 20 amino acids')
        
    df = get_cath.read_data('cath-domain-description-file.txt')
    filtered_df = get_cath.filter_with_user_list(df,dataset)
    df_with_sequence=get_cath.append_sequence(filtered_df,Path(path_to_assemblies),Path(path_to_pdb))
    get_cath.compare(df_with_sequence,path_to_models,path_to_dataset_map,acid_key,path_to_pdb,path_to_evoef,include,ignore_uncommon,by_fragment)

if __name__ == '__main__':
    compare_models()