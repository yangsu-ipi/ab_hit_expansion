import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from Levenshtein import distance
import sklearn.linear_model as lm
from sklearn.metrics import roc_curve, auc
import scipy.stats as stats
import sklearn


cdr3_alphabet = 'ACDEFGHIKLMNPQRSTVWY'

def normalize_abundance(df, col):
    s = df[col].fillna(0)
    s = (s / s.sum()) * 1e6
    s[s < 1] = 1
    return s

# def filter_rounds(df, cols):
#     return df[(df[cols] > 5).any(axis=1)]

def calc_enrichment(df, col1, col2, col1_min=None, col2_min=None):
    s1, s2 = normalize_abundance(df, col1), normalize_abundance(df, col2)
    enrichment = np.log(s2) - np.log(s1)
    if col1_min is not None and col2_min is not None:
        enrichment[(df[col1].fillna(0) < col1_min) & (df[col2].fillna(0) < col2_min)] = 0
    elif col1_min is not None:
        enrichment[df[col1].fillna(0) < col1_min] = 0
    elif col2_min is not None:
        enrichment[df[col2].fillna(0) < col2_min] = 0
#     enrichment[(df[[col1, col2]].fillna(0) < 2).all(axis=1)] = 0
    return enrichment

def filter_cdr3(seqs: pd.Series):
#     return s[:3] == "CAR" and s[-3:] in {"FDY", "LDY", "FDI", "FDP"}
    return (seqs.str[:3] == "CAR") & seqs.str[-3:].isin({"FDY", "LDY", "FDI", "FDP"})

def get_kmer_list(seq, include_framework=''):
    if 'C' in include_framework:
        seq = 'C' + seq
    if 'W' in include_framework:
        seq = seq + 'W'
    kmer_counts = {}

    kmer_len = 1
    num_chunks = (len(seq)-kmer_len)+1
    for idx in range(0,num_chunks):
        kmer = seq[idx:idx+kmer_len]
        assert len(kmer) == kmer_len
        if kmer in kmer_counts:
            kmer_counts[kmer] += 1
        else:
            kmer_counts[kmer] = 1

    kmer_len = 2
    num_chunks = (len(seq)-kmer_len)+1
    for idx in range(0,num_chunks):
        kmer = seq[idx:idx+kmer_len]
        assert len(kmer) == kmer_len
        if kmer in kmer_counts:
            kmer_counts[kmer] += 1
        else:
            kmer_counts[kmer] = 1

    kmer_len = 3
    num_chunks = (len(seq)-kmer_len)+1
    for idx in range(0,num_chunks):
        kmer = seq[idx:idx+kmer_len]
        assert len(kmer) == kmer_len
        if kmer in kmer_counts:
            kmer_counts[kmer] += 1
        else:
            kmer_counts[kmer] = 1
    #print kmer_counts
    return [(key,val) for key,val in kmer_counts.items()]

def get_kmer_to_idx():
    kmer_to_idx = {}
    kmer_list = [aa for aa in cdr3_alphabet]
    for aa in cdr3_alphabet:
        for bb in cdr3_alphabet:
            kmer_list.append(aa+bb)
            for cc in cdr3_alphabet:
                kmer_list.append(aa+bb+cc)

    kmer_to_idx = {aa: i for i, aa in enumerate(kmer_list)}
    return kmer_list, kmer_to_idx

def cdr3_seqs_to_arr(seqs, include_framework=''):
    kmer_list, kmer_to_idx = get_kmer_to_idx()
    seq_to_kmer_vector = {}
    for seq in seqs:
        # Make into kmers
        kmer_data_list = get_kmer_list(seq, include_framework=include_framework)
        norm_val = 0.
        for kmer,count in kmer_data_list:
            count = float(count)
            norm_val += (count * count)
        norm_val = np.sqrt(norm_val)

        # L2 normalize
        final_kmer_data_list = []
        for kmer,count in kmer_data_list:
            final_kmer_data_list.append((kmer_to_idx[kmer],float(count)/norm_val))

        # save to a dictionary
        seq_to_kmer_vector[seq] = final_kmer_data_list

    kmer_arr = np.zeros((len(seqs), len(kmer_to_idx)), dtype=np.float32)
    for i, seq in enumerate(seqs):
        kmer_vector = seq_to_kmer_vector[seq]
        for j_kmer,val in kmer_vector:
            kmer_arr[i, j_kmer] = val
    return kmer_arr

def train_model(output_dir, df, macs_col, facs_train_col, facs_val_col=""):
    params_file = os.path.join(output_dir, "kmer_LR_model_params.pkl")

    to_remove = (
        df['cdr3_aa'].str.contains(rf"[^{cdr3_alphabet}]") | 
        (df['vh_scaffold'] == "UNK") | 
        (df['vl_scaffold'] == "UNK")
    )
    df = df[~to_remove]
    print("Dropped", to_remove.sum(), "unusable leads")
    print(len(df), "usable leads")

    train_enrichment_col = f"{facs_train_col}_{macs_col}_enrichment"
    df[train_enrichment_col] = calc_enrichment(df, macs_col, facs_train_col)

    kmer_list, kmer_to_idx = get_kmer_to_idx()
    kmer_arr = cdr3_seqs_to_arr(df['cdr3_aa'], include_framework='W')
    # vh_onehot = pd.get_dummies(pd.Categorical(df['heavy'], categories=IPI_VH_SEQS_V2, ordered=True))
    # vl_onehot = pd.get_dummies(pd.Categorical(df['light'], categories=IPI_VL_SEQS, ordered=True))
    # length_onehot = pd.get_dummies(pd.Categorical(df['CDR3'].str.len(), ordered=True))
    vl_onehot = pd.get_dummies(df['vl_scaffold'])
    length_onehot = pd.get_dummies(df['cdr3_aa'].str.len())
    # print(vh_onehot.columns.values)
    print("VL scaffolds:", vl_onehot.columns.values.tolist())
    kmer_vh_vl_arr = np.concatenate([
        kmer_arr, 
    #     vh_onehot.values,
        vl_onehot.values
    ], axis=1)
    kmer_vh_vl_len_arr = np.concatenate([kmer_vh_vl_arr, length_onehot.values], axis=1)
    kmer_arr_labels = kmer_list
    kmer_vh_vl_arr_labels = (
        kmer_arr_labels +
    #     vh_onehot.columns.tolist() +
        vl_onehot.columns.tolist()
    )
    kmer_vh_vl_len_arr_labels = kmer_vh_vl_arr_labels + length_onehot.columns.tolist()
    

    X, X_labels, y = kmer_vh_vl_arr, kmer_vh_vl_arr_labels, df[train_enrichment_col].values
    X = pd.DataFrame(X, columns=X_labels)
    X, y = X[y != 0], y[y != 0]
    np.random.seed(1)
    msk = np.random.permutation(len(y)) < int(len(y) * 0.8)
    X_train, X_test = X[msk], X[~msk]
    y_train, y_test = y[msk], y[~msk]
    print("Train set:", len(y_train), "leads,", (y_train > 0).sum(), "enriched")
    print("Test set:", len(y_test), "leads,", (y_test > 0).sum(), "enriched")

    # train model
    thresh = 0.
    if os.path.exists(params_file) and False:
        with open(params_file, "rb") as f:
            clf = pickle.load(f)
    else:
        clf = lm.LogisticRegression(random_state=42, penalty='l1', C=1., class_weight='balanced', solver='liblinear').fit(X_train, y_train > thresh)
        with open(params_file, "wb") as f:
            pickle.dump(clf, f)

    y_score_train = clf.predict_proba(X_train)[:, 1]
    fpr_train, tpr_train, _ = roc_curve(y_train > 0, y_score_train)
    roc_auc_train = auc(fpr_train, tpr_train)
    print(f"Train AUC: {roc_auc_train:.2f}")

    y_score = clf.predict_proba(X_test)[:, 1]
    fpr_test, tpr_test, _ = roc_curve(y_test > 0, y_score)
    roc_auc_test = auc(fpr_test, tpr_test)
    print(f"Test AUC: {roc_auc_test:.2f}")

    X, X_labels, y = kmer_vh_vl_arr, kmer_vh_vl_arr_labels, df[train_enrichment_col].values
    X = pd.DataFrame(X, columns=X_labels)
    df["LR_score"] = clf.predict_proba(X)[:, 1]

    if facs_val_col != "":
        # aff3_file = "validation_scores.csv"
    
        val_enrichment_col = f"{facs_val_col}_{macs_col}_enrichment"
        df[val_enrichment_col] = calc_enrichment(df, macs_col, facs_val_col)

        X_val, y_val = kmer_vh_vl_arr, df[val_enrichment_col].values
        X_val = pd.DataFrame(X_val, columns=X_labels)
        X_val, y_val = X_val[y_val != 0], y_val[y_val != 0]
        print("Validation set:", len(y_val), "leads,", (y_val > 0).sum(), "enriched")

        y_score_val = clf.predict_proba(X_val)[:, 1]
        fpr_val, tpr_val, _ = roc_curve(y_val > 0, y_score_val)
        roc_auc_val = auc(fpr_val, tpr_val)
        print(f"Validation AUC: {roc_auc_val:.2f}")

        X, X_labels, y = kmer_vh_vl_arr, kmer_vh_vl_arr_labels, df[facs_val_col].values
        X = pd.DataFrame(X, columns=X_labels)
    #     X, y = X[y > 0], y[y > 0]
        y_score = clf.predict_proba(X)[:, 1]
        spearman_val = stats.spearmanr(y_score[y > 0], y[y > 0])[0]
        print(f"{facs_val_col} Spearman: {spearman_val:.2f}")
    #     save_df = df.copy()
    # #     save_df = save_df[save_df["Aff3"] > 0]
    #     save_df["LR_score"] = y_score
    #     save_df.to_csv(aff3_file, index=False)

    
    print("Top k-mers")
    print("----------")
    coefs = pd.Series(index=clf.feature_names_in_, data=clf.coef_[0])
    print(coefs[(coefs - coefs.mean()).abs() >= 10 * coefs.std()].sort_values(ascending=False))
    # print(coefs.mean(), coefs.std())
    # print(coefs[coefs != 0].sort_values(ascending=False))
    # print(coefs[coefs != 0])

    fig, ax = plt.subplots()
    ax.set_title("ROC Curve")
    ax.plot(fpr_test, tpr_test, 'C0', label = f'Test AUC = {roc_auc_test:0.2f}')
    ax.plot(fpr_val, tpr_val, 'C1', label = f'Validation AUC = {roc_auc_val:0.2f}')
    ax.legend(loc = 'lower right')
    ax.plot([0, 1], [0, 1],'r--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    fig.savefig(os.path.join(output_dir, "00_roc_curve.png"))
    plt.close(fig)
    
    # fig, ax = plt.subplots()
    # plt.title(f"{target} SPR")
    # plt.scatter(char_df["SPR KD"].mask(lambda col: col > 300, 300), y_score_val2)
    # ax.invert_xaxis()
    # plt.ylabel("model score")
    # plt.xlabel("SPR KD (nM)")
    # fig.canvas.draw()
    # labels = [item.get_text() for item in ax.get_xmajorticklabels()]
    # labels = ["Fail" if item == "300" else item for item in labels]
    # ax.set_xticks(ax.get_xticks(minor=False)[1:-1])
    # ax.set_xticklabels(labels[1:-1])
    # plt.show()

    # fig, ax = plt.subplots()
    # plt.title(f"{target} Cell Display")
    # plt.scatter(char_df["Cell Display EC50"].mask(lambda col: col > 300, 300), y_score_val2)
    # plt.gca().invert_xaxis()
    # plt.ylabel("model score")
    # plt.xlabel("EC50 (nM)")
    # fig.canvas.draw()
    # labels = [item.get_text() for item in ax.get_xmajorticklabels()]
    # labels = ["Fail" if item == "300" else item for item in labels]
    # ax.set_xticks(ax.get_xticks(minor=False)[1:-1])
    # ax.set_xticklabels(labels[1:-1])
    # plt.show()

    return df

def min_levenshtein(seq1, seqs):
    return min(distance(seq1, s, score_cutoff=len(seq1)) for s in seqs)

def min_pairwise_levenshtein(seqs):
    prev_cdr3s = [""]
    distances = []
    for cdr3 in seqs:
        distances.append(min_levenshtein(cdr3, prev_cdr3s))
        prev_cdr3s.append(cdr3)
    return distances

def select_hits(output_dir, df, facs_train_col, min_aff1_frac=1/5000, min_lr_score=0.8, min_dist_to_ordered=5, min_pairwise_dist=5):
    subset_df = df[(df[facs_train_col] > df[facs_train_col].sum() * min_aff1_frac) & (df["LR_score"] > min_lr_score)].sort_values(by="LR_score", ascending=False)
    # subset_df["min_dist_to_ordered"] = subset_df["cdr3_aa"].apply(min_levenshtein, args=(ordered_set["CDR3"],))
    # subset_df = subset_df[subset_df["min_dist_to_ordered"] >= min_dist_to_ordered]

    subset_df["min_pairwise_dist"] = min_pairwise_levenshtein(subset_df["cdr3_aa"].values)
    subset_df = subset_df[subset_df["min_pairwise_dist"] >= min_pairwise_dist]
    subset_df.to_csv(os.path.join(output_dir, "final_clones_for_synthesis.csv"), index=False)
    return subset_df


