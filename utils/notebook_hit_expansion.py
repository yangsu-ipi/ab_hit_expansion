from . import kmer_regession_hit_expansion_ngs as N
from . import kothiwal_kollasch as K
import pandas as pd
import os
from glob import glob
import shutil
from ipyfilechooser import FileChooser
from ipywidgets import Dropdown, Label, Image, HBox, VBox, Text, Output
from IPython.display import display, HTML

class HitExpanssionBase():
    def run(self):
        raise NotImplementedError()

    def display_results(self):
        raise NotImplementedError()

class HitExpansionNguyen(HitExpanssionBase):
    def __init__(self, input_file, input_df, output_folder, macs_col, facs_col, val_facs_col, use_blosum_in_model):
        self.INPUT_FILE = input_file
        self.input_df = input_df
        # OUTPUT_FOLDER = os.path.splitext(INPUT_FILE)[0] + "_hit_expansion"
        self.OUTPUT_FOLDER = output_folder

        self.USE_BLOSUM_IN_MODEL = use_blosum_in_model
        self.macs_col = macs_col
        self.facs_col = facs_col
        self.val_facs_col = val_facs_col

        self.TRAINING_MODE = "binary_strong"
        self.USE_MACS_AS_GERMLINE = True

        self.DIVERSITY_METRIC = "levenshtein"
        self.MIN_LEVENSHTEIN_DIST = 5

        self.PLOT_TOP_N_CLUSTER = 50
        self.LABEL_DENDROGRAM_WITH_CDR3 = True

        self.PREVIOUS_FILES = [] # "/Users/Hoan.Nguyen/ComBio/AntigenDB/datasources/ipi_data/processed/ipi_antibodydb_july2025.csv"
        self.PREVIOUS_CDR3_COLUMN = "CDR3"

        self.kmer_logreg_score_df = None
        self.model = None
        self.X = None
        self.leads_df = None

    def run(self):
        self.kmer_logreg_score_df, self.model, self.X = N.add_kmer_logreg_score(self.input_df, use_blosum_features=self.USE_BLOSUM_IN_MODEL, training_mode=self.TRAINING_MODE, cdr3_col='cdr3_aa', macs_col=self.macs_col, facs1_col=self.facs_col)
        previous_list = N.load_previous_cdr3s(self.PREVIOUS_FILES, self.PREVIOUS_CDR3_COLUMN)
        self.leads_df = N.select_diverse_leads(
            self.kmer_logreg_score_df, 
            previous_cdr3s=previous_list, 
            score_col=self.kmer_logreg_score_df.columns[-1], 
            diversity_metric=self.DIVERSITY_METRIC, 
            cdr3_col='cdr3_aa',
            count_col=self.facs_col, 
            min_levenshtein_dist=self.MIN_LEVENSHTEIN_DIST
        )
        N.PLOT_TOP_N_CLUSTER = self.PLOT_TOP_N_CLUSTER
        N.LABEL_DENDROGRAM_WITH_CDR3 = self.LABEL_DENDROGRAM_WITH_CDR3
        N.generate_evaluation_plots(self.leads_df, self.leads_df.columns[-2], self.TRAINING_MODE, self.OUTPUT_FOLDER, model=self.model, X=self.X, cdr3_col='cdr3_aa', macs_col=self.macs_col, facs1_col=self.facs_col)

        # Final CSVs
        self.leads_df.to_csv(os.path.join(self.OUTPUT_FOLDER, "leads_with_ml_score_and_selection.csv"), index=False)
        self.leads_df[self.leads_df["selected_for_synthesis"]].to_csv(os.path.join(self.OUTPUT_FOLDER, "final_clones_for_synthesis.csv"), index=False)

    def display_results(self):
        df = pd.read_csv(os.path.join(self.OUTPUT_FOLDER, "final_clones_for_synthesis.csv"))
        df = df[["cdr3_aa", "vh_scaffold", "vl_scaffold", "cdr3_functional", "log_fold_change", "kmer_logreg_score"]]
        df.rename(columns={"kmer_logreg_score": "score"}, inplace=True)

        # output_widget = Output()
        # with output_widget:
        #     display(HTML(df.to_html()))
        # display(VBox([output_widget] + list(images.values())))

        display(HTML(df.to_html()))

        images = {}
        for fn in sorted(glob(os.path.join(self.OUTPUT_FOLDER, "*.png"))):
            with open(fn, "rb") as f:
                images[os.path.splitext(os.path.basename(fn))[0]] = Image(value=f.read(), format='png', width=800) #, height=400
        
        display(VBox(tuple(images.values())))
        # for img in images.values():
        #     display(img)

        
class HitExpansionKothiwalKollasch(HitExpanssionBase):
    def __init__(self, input_file, input_df, output_folder, macs_col, facs_col, val_facs_col):
        self.INPUT_FILE = input_file
        self.input_df = input_df
        # OUTPUT_FOLDER = os.path.splitext(INPUT_FILE)[0] + "_hit_expansion"
        self.OUTPUT_FOLDER = output_folder

        self.macs_col = macs_col
        self.facs_col = facs_col
        self.val_facs_col = val_facs_col

        self.min_aff1_frac = 1/5000
        self.min_lr_score = 0.8
        self.min_dist_to_ordered = 5 
        self.min_pairwise_dist = 5

    def run(self):
        score_df = K.train_model(self.OUTPUT_FOLDER, self.input_df, self.macs_col, self.facs_col, self.val_facs_col)
        hits_df = K.select_hits(
            self.OUTPUT_FOLDER, 
            score_df, 
            self.facs_col, 
            min_aff1_frac = self.min_aff1_frac, 
            min_lr_score = self.min_lr_score, 
            min_dist_to_ordered = self.min_dist_to_ordered, 
            min_pairwise_dist = self.min_pairwise_dist, 
        )

    def display_results(self):
        df = pd.read_csv(os.path.join(self.OUTPUT_FOLDER, "final_clones_for_synthesis.csv"))
        enrichment_cols = [col for col in df.columns if col.endswith("_enrichment")]
        df = df[["cdr3_aa", "vh_scaffold", "vl_scaffold", "cdr3_functional"] + enrichment_cols + ["LR_score"]]
        df.rename(columns={"LR_score": "score"}, inplace=True)

        # output_widget = Output()
        # with output_widget:
        #     display(HTML(df.to_html()))
        # display(VBox([output_widget] + list(images.values())))

        display(HTML(df.to_html()))

        images = {}
        for fn in sorted(glob(os.path.join(self.OUTPUT_FOLDER, "*.png"))):
            with open(fn, "rb") as f:
                images[os.path.splitext(os.path.basename(fn))[0]] = Image(value=f.read(), format='png', width=800) #, height=400
        
        display(VBox(tuple(images.values())))
        # for img in images.values():
        #     display(img)


class HitExpansionUI():
    def __init__(
        self, 
        input_root = os.path.expanduser("~"), 
        output_root = os.path.expanduser("~"), 
    ):
        self.input_root = input_root
        self.output_root = output_root
        
        self.INPUT_FILE = ""
        self.input_df = None
        self.ngs_rounds = {}
        # OUTPUT_FOLDER = os.path.splitext(INPUT_FILE)[0] + "_hit_expansion"
        self.OUTPUT_FOLDER = ""

        self.features = "CDRH3 kmers + VL"
        self.macs_col = ""
        self.facs_col = ""
        self.val_facs_col = ""

        self.ddEarlierRound = Dropdown(
            options=[],
            description='Earlier Round',
            layout={'width': 'max-content'}
        )
        self.ddEarlierRound.observe(self.on_earlier_round_change, names='value')

        self.ddLaterRound = Dropdown(
            options=[],
            description='Later Round', 
            layout={'width': 'max-content'}
        )
        self.ddLaterRound.observe(self.on_later_round_change, names='value')

        self.ddValidationRound = Dropdown(
            options=[],
            description='Validation Round', 
            layout={'width': 'max-content'}
        )
        self.ddValidationRound.observe(self.on_validation_round_change, names='value')

        self.ddFeatures = Dropdown(
            value=self.features,
            options=["CDRH3 kmers + VL", "CDRH3 kmers", "CDRH3 kmers + BLOSUM62"],
            description='Features', 
            layout={'width': 'max-content'}
        )
        self.ddFeatures.observe(self.on_features_change, names='value')

        # self.lblOutputPath = Label(value="Output Location:", layout={'width': 'max-content'})
        # self.txtOutputPath = Text(value="", disabled=True, layout={'width': 'max-content'})

        self.fc = FileChooser(self.input_root)
        self.fc.sandbox_path = self.input_root
        self.fc.filter_pattern = '*_clones.csv'
        self.fc.register_callback(self.on_file_change)


    def on_earlier_round_change(self, change):
        self.macs_col = self.ngs_rounds[change.new]

    def on_later_round_change(self, change):
        self.facs_col = self.ngs_rounds[change.new]

    def on_validation_round_change(self, change):
        self.val_facs_col = self.ngs_rounds[change.new]

    def on_file_change(self, chooser):
        self.INPUT_FILE = chooser.selected
        self.input_df = pd.read_csv(self.INPUT_FILE)
        count_cols = self.input_df.columns[self.input_df.columns.str.startswith("count ")].tolist()
        prefix = os.path.commonprefix(count_cols)
        self.ngs_rounds = {"": ""} | {c[len(prefix):]: c for c in count_cols}
        self.ddEarlierRound.options = list(self.ngs_rounds.keys())
        self.ddEarlierRound.value = ""
        self.ddLaterRound.options = list(self.ngs_rounds.keys())
        self.ddLaterRound.value = ""
        self.ddValidationRound.options = list(self.ngs_rounds.keys())
        self.ddValidationRound.value = ""
        
        self.OUTPUT_FOLDER = os.path.join(self.output_root, os.path.splitext(self.INPUT_FILE)[0].lstrip("/") + "_hit_expansion")
        # self.txtOutputPath.value = self.OUTPUT_FOLDER


    def on_features_change(self, change):
        self.features = change.new

    def setup(self):
        display(VBox([
            self.fc, 
            self.ddEarlierRound, 
            self.ddLaterRound, 
            self.ddValidationRound, 
            self.ddFeatures, 
            # HBox([self.lblOutputPath, self.txtOutputPath])
        ], layout={'width': 'max-content'}))

    def run(self):
        if os.path.isdir(self.OUTPUT_FOLDER):
            shutil.rmtree(self.OUTPUT_FOLDER)
            os.mkdir(self.OUTPUT_FOLDER)

        print("Output location:", self.OUTPUT_FOLDER)
        if (self.features == "CDRH3 kmers") or (self.features == "CDRH3 kmers + BLOSUM62"):
            he = HitExpansionNguyen(
                input_file = self.INPUT_FILE, 
                input_df = self.input_df, 
                output_folder = self.OUTPUT_FOLDER, 
                macs_col = self.macs_col, 
                facs_col = self.facs_col, 
                val_facs_col = self.val_facs_col, 
                use_blosum_in_model = self.features == "CDRH3 kmers + BLOSUM62"
            )
        elif self.features == "CDRH3 kmers + VH + VL":
            he = HitExpansionKothiwalKollasch(
                input_file = self.INPUT_FILE, 
                input_df = self.input_df, 
                output_folder = self.OUTPUT_FOLDER, 
                macs_col = self.macs_col, 
                facs_col = self.facs_col, 
                val_facs_col = self.val_facs_col, 
            )
        else:
            print("FEATURES:", self.features)

        he.run()
        he.display_results()
