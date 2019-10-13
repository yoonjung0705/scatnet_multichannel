import os
import re
import glob

def match_filename(pattern, root_dir):
    '''returns list of matched patterns in file names

    inputs
    ------
    pattern - raw text indicating the regex pattern to use for the file names
        The part in the string to extract should be denoted with parentheses ()
    root_dir - directory to search the files

    outputs
    matches - list of strings that are file names that match the given pattern
    -------
    
    '''
    file_names = glob.glob(os.path.join(root_dir, "*"))
    matches = [re.fullmatch(pattern, os.path.basename(file_name)) for file_name in file_names]
    matches = [match.group(1) for match in matches if match is not None]
    
    return matches
    

    
