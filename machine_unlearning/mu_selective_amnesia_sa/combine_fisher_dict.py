def combine_fisher_dict():
    
    import os
    import pickle
    import glob
    matching_files = glob.glob("fisher_dict_rank*")
    assert len(matching_files) > 0
    
    with open(matching_files[0], 'rb') as f:
        fisher_dict = pickle.load(f)
        os.remove(matching_files[0])
    
    for file in matching_files[1:]:
        with open(file, 'rb') as f:
            fisher_dict_temp = pickle.load(f)
        for key in fisher_dict:
            fisher_dict[key] += fisher_dict_temp[key]
        os.remove(file)
        
    with open("full_fisher_dict.pkl", 'wb') as f:
        pickle.dump(fisher_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
if __name__ == "__main__":
    combine_fisher_dict()