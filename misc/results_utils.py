import pandas as pd
import numpy as np

def df_to_latex(expnam, cols_to_report=["Success", "Avg Success","L2 Distance","Diversity"]):
    df_li = []
    datasets = ["creditdefault","givecredit","adultcensus"]
    base_models = ["LogisticRegressionSK","RandomForestSK","BaselineDNN"]
    rc = "StEP"
    
    max_err= 0
    csvnames = ["partial_failed_agg_all_trials_results.csv", "partial_failed_agg_all_trials_err.csv"]
    results = []
    for csvname in csvnames:
        df_li = []
        for i_d, d in enumerate(datasets):
            df_li_d = []
            for rc in ["StEP","DiCE","FACE","CCHVAE"]:    
                df_li_b = []
                for i,b in enumerate(base_models):
                    expnam1 = f"results/"+ expnam
                    df_result = pd.read_csv(expnam1 + f"/{d}/{b}/{rc}/"+csvname)
                    cols_to_drop = ["poi_id","min_l2_path_len","min_l2_prox","min_l2_path_steps","max_l2_path_len","max_l2_prox","max_l2_path_steps"]
                    df_new = df_result.copy()
                    df_new = df_new.drop(columns=cols_to_drop)
                    df_new = df_new.rename(columns={"pois_with_geq_1_success": "Success", "path_successes": "Avg Success", "l2_prox": "L2 Distance","diversity": "Diversity", "n_neg_samples":"n Samples","l2_path_len": "Path Length","l2_path_steps":"Path Steps"})

                    if rc == "StEP":
                        df_new["Dataset"] = d
                    else:
                        df_new["Dataset"] = None
                    if i == 0:
                        df_new["Method"] = rc
                        df = df_new[["Dataset","Method"]+cols_to_report]
                    else:
                        df = df_new[cols_to_report]
                    df_new["Dataset"] = d
                    df_new["Method"] = rc
                    df_new["Model"] = b
                    df = df_new[["Dataset","Model","Method"]+cols_to_report]
                    df_li_b.append(df)

                df = pd.concat(df_li_b,ignore_index=True)
                
                df_li_d.append(df)
            df = pd.concat(df_li_d,ignore_index=True)
            df1 = df.pivot_table(index=["Dataset","Model"], columns = ["Method"], aggfunc='sum', sort = False)
            
            df['Dataset'].loc[df['Method'] == "StEP"] = d
            df_li.append(df)
        df = pd.concat(df_li,ignore_index=True)
        
        #df.to_csv(f'results/10trial_latex_ready.csv',float_format="%.2f",index=False)
        df1 = df.pivot_table(index=["Dataset","Method"], columns="Model", values = cols_to_report, aggfunc='sum', sort = False)
        df1 = df1.swaplevel(0, 1, axis=1).sort_index(axis=1)
        cols = ['LogisticRegressionSK', 'RandomForestSK', 'BaselineDNN']
        new_cols = df1.columns.reindex(cols, level=0)
        df1 = df1.reindex(columns=new_cols[0])
        df1 = df1.reindex(cols_to_report, level=1, axis=1)
        print(df1.to_latex())
        print(max_err)
        results.append(df1)
    for r in results:
        print(r.round(decimals=2))
        print(r.round(decimals=2).to_latex())
    print((results[1]/results[0]).max(axis=0).round(decimals=4))
    print((results[1]/results[0]).max(axis=1).round(decimals=4))

def df_to_latex_noise(expnam1,expnam2, cols_to_report=["Success", "Avg Success","L2 Distance","Diversity"]):
    df_li = []
    datasets = ["creditdefault","givecredit","adultcensus"]
    base_models = ["LogisticRegressionSK","RandomForestSK","BaselineDNN"]
    rc = "StEP"
    csvnames = ["partial_failed_agg_all_trials_results.csv", "partial_failed_agg_all_trials_err.csv"]
    results = []
    for csvname in csvnames:
        df_li = []
        for i_d, d in enumerate(datasets):
            df_li_d = []
            for noise in [0.0,0.1,0.3,0.5]:
                noise = round(noise, 1)    
                df_li_b = []
                for i,b in enumerate(base_models):
                    expnam = f"results/"+ expnam1+ str(noise)+expnam2
                    df_result = pd.read_csv(expnam + f"/{d}/{b}/{rc}/"+csvname)
                
                    cols_to_drop = ["poi_id","min_l2_path_len","min_l2_prox","min_l2_path_steps","max_l2_path_len","max_l2_prox","max_l2_path_steps"]
                    df_new = df_result.copy().drop(columns=cols_to_drop)
                    df_new = df_new.rename(columns={"pois_with_geq_1_success": "Success", "path_successes": "Avg Success", "l2_prox": "L2 Distance","diversity": "Diversity", "n_neg_samples":"n Samples","l2_path_len": "Path Length","l2_path_steps":"Path Steps"})
                    
                    if noise == 0.0:
                        df_new["Dataset"] = d
                    else:
                        df_new["Dataset"] = None
                    if i == 0:
                        df_new["Noise"] = noise
                        df = df_new[["Dataset","Noise"]+cols_to_report]
                    else:
                        df = df_new[cols_to_report]
                    df_new["Dataset"] = d
                    df_new["Noise"] = noise
                    df_new["Model"] = b
                    df = df_new[["Dataset","Model","Noise"]+cols_to_report]
                    df_li_b.append(df)

                df = pd.concat(df_li_b,ignore_index=True)
                df_li_d.append(df)
            df = pd.concat(df_li_d,ignore_index=True)
            df1 = df.pivot_table(index=["Dataset","Model"], columns = ["Noise"], aggfunc='sum', sort = False)
            
            df['Dataset'].loc[df['Noise'] == 0.0] = d
            df_li.append(df)
        df = pd.concat(df_li,ignore_index=True)
        
        #df.to_csv(f'results/10trial_latex_ready.csv',float_format="%.2f",index=False)
        print(df)
        df1 = df.pivot_table(index=["Dataset","Noise"], columns="Model", values = cols_to_report, aggfunc='sum', sort = False)
        print(df1)
        df1 = df1.swaplevel(0, 1, axis=1).sort_index(axis=1)
        cols = ['LogisticRegressionSK', 'RandomForestSK', 'BaselineDNN']
        new_cols = df1.columns.reindex(cols, level=0)
        df1 = df1.reindex(columns=new_cols[0])
        df1 = df1.reindex(cols_to_report, level=1, axis=1)
        print(df1)
        print(df1.to_latex())
        results.append(df1)
    for r in results:
        print(r.copy().round(decimals=2))
        print(r.copy().round(decimals=2).to_latex())
    print((results[1]/results[0]).max(axis=0).round(decimals=4))
    print((results[1]/results[0]).max(axis=1).round(decimals=4))