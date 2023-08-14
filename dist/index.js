importScripts("https://cdn.jsdelivr.net/pyodide/v0.23.4/pyc/pyodide.js");

function sendPatch(patch, buffers, msg_id) {
  self.postMessage({
    type: 'patch',
    patch: patch,
    buffers: buffers
  })
}

async function startApplication() {
  console.log("Loading pyodide!");
  self.postMessage({type: 'status', msg: 'Loading pyodide'})
  self.pyodide = await loadPyodide();
  self.pyodide.globals.set("sendPatch", sendPatch);
  console.log("Loaded!");
  await self.pyodide.loadPackage("micropip");
  const env_spec = ['https://cdn.holoviz.org/panel/1.2.1/dist/wheels/bokeh-3.2.1-py3-none-any.whl', 'https://cdn.holoviz.org/panel/1.2.1/dist/wheels/panel-1.2.1-py3-none-any.whl', 'pyodide-http==0.2.1', 'hvplot', 'pandas']
  for (const pkg of env_spec) {
    let pkg_name;
    if (pkg.endsWith('.whl')) {
      pkg_name = pkg.split('/').slice(-1)[0].split('-')[0]
    } else {
      pkg_name = pkg
    }
    self.postMessage({type: 'status', msg: `Installing ${pkg_name}`})
    try {
      await self.pyodide.runPythonAsync(`
        import micropip
        await micropip.install('${pkg}');
      `);
    } catch(e) {
      console.log(e)
      self.postMessage({
	type: 'status',
	msg: `Error while installing ${pkg_name}`
      });
    }
  }
  console.log("Packages loaded!");
  self.postMessage({type: 'status', msg: 'Executing code'})
  const code = `
  
import asyncio

from panel.io.pyodide import init_doc, write_doc

init_doc()

import panel as pn
import json
import pandas as pd
from io import StringIO
import hvplot.pandas  # Enable interactive

pn.extension(sizing_mode="stretch_width", template="material")

DATE = "August 14, 2023"
TARGET_REPO = "scikit-learn/scikit-learn"
TRACKER_REPO = "thomasjpfan/sklearn-pr-tracker"

pn.state.template.param.update(
    site=TARGET_REPO,
    title="Pull Requests Overview",
    header_background="#520b92",
)

FILE = StringIO(
    r"""
pr,title,additions,deletions,comments,approvals,updated
2630,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/2630"">Make Pipeline compatible with AdaBoost</a>",85,25,36,0,2021-01-22 10:48:13+00:00
2738,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/2738"">Add user option to use SVD for orthogonalizing the mixing matrix in FastICA</a>",50,13,13,0,2021-01-22 10:48:13+00:00
3126,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/3126"">Updated K-means clustering for Nystroem</a>",221,70,23,0,2021-01-22 10:48:16+00:00
3342,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/3342"">Truncating in MinMaxScaler</a>",65,2,12,0,2023-07-30 09:49:47+00:00
3383,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/3383"">[WIP] Ridge path</a>",705,137,69,0,2021-12-06 21:47:27+00:00
3529,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/3529"">ENH: Add a default label to LabelPropagation algorithms (fixes #3218)</a>",46,4,14,0,2023-07-30 10:07:58+00:00
3739,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/3739"">[MRG] K-SVD</a>",544,121,57,0,2021-01-22 10:48:21+00:00
3907,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/3907"">[WIP] Adding tests for estimators implementing \`partial_fit\` and a few other related fixes / enhancements</a>",237,67,288,0,2023-07-29 12:22:06+00:00
4087,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/4087"">[MRG] support sample_weight in silhouette_score</a>",119,69,14,0,2021-01-22 10:48:29+00:00
4215,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/4215"">[MRG] GBM & meta-ensembles - support for class_weight</a>",337,25,6,0,2022-02-14 02:24:50+00:00
4220,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/4220"">[WIP] Add warm_start to Ridge. </a>",58,10,2,1,2022-07-29 13:17:23+00:00
4288,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/4288"">[WIP] Allow nan for userdefined metric in dbscan et al</a>",113,43,18,0,2021-04-13 15:34:54+00:00
4289,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/4289"">Mutual Information estimator based on the Renyi quadratic entropy and th...</a>",958,0,14,0,2021-01-22 10:48:33+00:00
4301,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/4301"">[MRG] Choose number of clusters</a>",1149,0,107,0,2021-01-22 10:48:34+00:00
4501,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/4501"">[MRG] Add distance threshold on Hierarchical Clustering, see #3796</a>",86,8,13,0,2021-01-22 10:48:35+00:00
4515,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/4515"">Added new kernels for Mean Shift clustering</a>",236,20,24,0,2022-10-04 20:45:54+00:00
4522,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/4522"">[WIP] Metrics testing</a>",29,11,6,0,2021-01-22 10:48:36+00:00
4693,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/4693""> making dictionary learning closer to the SparseNet algorithm</a>",565,2,31,0,2021-01-22 10:48:37+00:00
4703,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/4703"">[MRG] RandomActivation</a>",608,1,86,0,2021-01-22 10:48:38+00:00
4822,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/4822"">[MRG] ROCCH calibration method</a>",254,23,16,0,2021-01-22 10:48:39+00:00
4974,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/4974"">Weighted pls -- adding support for sample_weight option.</a>",244,89,21,0,2021-01-22 10:48:43+00:00
5123,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/5123"">[Discussion] Sparse datastructure for MultinomialNB feature_count_</a>",157,19,6,0,2021-01-22 10:48:44+00:00
5287,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/5287"">[MRG] Implement FABIA biclustering algorithm</a>",577,1,3,0,2021-01-22 10:48:48+00:00
5312,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/5312"">[MRG+1] check_X_y should copy y also if copy is set to be True</a>",22,10,18,0,2021-01-22 10:48:50+00:00
5501,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/5501"">[WIP] Infomax ica</a>",816,0,7,0,2022-08-09 22:26:05+00:00
5532,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/5532"">[WIP] Add \`return_std\` option to ensembles</a>",224,24,26,0,2021-01-22 10:48:51+00:00
5593,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/5593"">[MRG] ENH: Support threshold='auto' in Birch</a>",38,2,17,0,2021-02-12 14:14:40+00:00
5826,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/5826"">New estimator: Rakel</a>",4284,2,8,0,2022-04-15 02:36:02+00:00
5972,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/5972"">Stratifying Across Classes During Training in ShuffleSplit #5965</a>",92,19,16,0,2021-01-22 10:48:54+00:00
6077,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/6077"">changed pls_.py to properly scale data, modified documents</a>",11,13,24,0,2021-01-22 10:48:55+00:00
6085,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/6085"">[WIP] randomised block krylov svd</a>",334,18,6,0,2021-01-22 10:48:56+00:00
6160,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/6160"">[MRG] Refactor model_selection._search to include transductive estimators</a>",1040,31,81,0,2021-01-22 10:48:56+00:00
6215,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/6215"">[WIP] Storing the best attributes of (non-GridSearch) CV models</a>",148,29,19,0,2021-01-22 10:48:57+00:00
6236,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/6236"">Better example for agglomerative clustering</a>",65,0,20,0,2021-12-10 00:02:30+00:00
6303,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/6303"">WIP: Nearest Neighbor chaining for ward</a>",131,55,2,0,2021-01-22 10:48:58+00:00
6338,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/6338"">Adding a utility function for plotting decision regions of classifiers</a>",167,0,45,0,2021-01-22 10:48:58+00:00
6448,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/6448"">[MRG] Support multi-threading of LibLinear L1 one-vs-rest LogisticRegression for # classes > 2</a>",746,160,5,0,2022-02-14 02:37:42+00:00
6534,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/6534"">[MRG] Add Information Gain and Information Gain Ratio feature selection functions</a>",528,40,95,0,2021-01-22 10:49:02+00:00
6540,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/6540"">[WIP] Add ensemble selection algorithm</a>",264,1,59,0,2021-01-22 10:49:03+00:00
6558,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/6558"">[MRG]Added possibility of adding both values and proportions in export_graphviz</a>",62,14,26,0,2022-01-15 02:21:09+00:00
6598,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/6598"">[MRG] Stratifiedkfold continuous (fixed)</a>",534,27,35,0,2021-01-22 10:49:04+00:00
6604,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/6604"">[MRG] LabelPropagation and LabelSpreading enhancements</a>",78,17,3,0,2022-02-14 02:38:47+00:00
6806,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/6806"">[WIP] Robust PCA algorithm</a>",393,1,21,0,2021-05-15 18:05:24+00:00
6832,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/6832"">[MRG] Documentation and some input validation for get_scorer</a>",74,8,11,0,2021-01-22 10:49:07+00:00
6930,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/6930"">[MRG] Use logging.info instead of print (#6929)</a>",13,10,24,0,2021-01-22 10:49:08+00:00
6948,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/6948"">[WIP] ENH Optimal n_clusters values</a>",982,1,27,0,2021-04-14 19:19:12+00:00
6992,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/6992"">ENH Isomap: adding choice option for neighborhood function (plus some code cleaning)</a>",77,25,9,0,2022-04-15 02:34:43+00:00
7078,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/7078"">[MRG] addressing issue #6887, adding callable support for covariance comp.</a>",64,3,15,0,2021-11-30 02:49:18+00:00
7107,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/7107"">Refactored Count Vectorizer to be more memory efficient on N-grams</a>",74,5,7,0,2021-01-22 10:49:11+00:00
7153,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/7153"">[WIP] Make neighbor tree module supports fused types</a>",120,41,13,0,2022-07-29 12:47:33+00:00
7178,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/7178"">[MRG] ENH support multiclass column vector array-like in metrics</a>",76,61,8,0,2021-01-22 10:49:12+00:00
7284,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/7284"">[DOC] GSoC BayesianGaussianMixture formula</a>",642,2,32,1,2021-12-16 23:12:22+00:00
7337,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/7337"">Fix feature importances computation error in RandomForest, ExtraTrees…</a>",12,9,2,0,2021-06-29 15:29:46+00:00
7383,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/7383"">[MRG] Added flag to disable l2-dist finite check</a>",262,51,45,0,2022-03-30 21:06:55+00:00
7761,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/7761"">[WIP] replaced n_iter by max_iter and added deprecation</a>",71,51,6,0,2021-01-22 10:49:16+00:00
7773,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/7773"">[MRG+½] label_binarize also for binary case</a>",74,51,78,1,2021-01-22 10:49:16+00:00
7829,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/7829"">Add Angular distance, a metric version of cosine distance</a>",23,2,26,0,2022-08-04 08:04:33+00:00
7839,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/7839"">[MRG+2-1] Add floating point option to max_feature option in CountVectorizer and TfidfVectorizer to use a percentage of the features.</a>",51,18,41,1,2021-01-22 10:49:17+00:00
7910,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/7910"">[MRG+1] Support Vector Data Description</a>",1093,143,191,1,2023-07-06 10:56:47+00:00
7916,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/7916"">[WIP] Gaussian Bernoulli RBM continuation of #2680</a>",438,168,3,0,2021-01-22 10:49:19+00:00
7996,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/7996"">[MRG] Tests for no sparse y support in RandomForests</a>",97,48,32,0,2021-01-22 10:49:20+00:00
8021,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/8021"">[MRG+1] Intersection/Jensen-Shannon kernel samplers</a>",318,88,93,0,2021-12-15 04:31:05+00:00
8030,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/8030"">[WIP] Add SmartSplitter for handling categorical features in decision tree</a>",517,47,3,0,2021-04-13 15:34:47+00:00
8105,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/8105"">[WIP] Progress Logger</a>",704,24,2,0,2022-02-14 02:39:07+00:00
8206,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/8206"">[WIP] Add prediction strength method to determine number of clusters</a>",809,66,28,0,2021-11-30 21:17:57+00:00
8230,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/8230"">[MRG] GridSearchCV.use_warm_start parameter for efficiency</a>",533,19,87,0,2022-04-02 23:41:02+00:00
8317,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/8317"">[WIP] ENH arbitrary diagnostic_func callback in GridSarchCV</a>",152,20,2,0,2021-01-22 10:49:23+00:00
8350,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/8350"">[MRG] FIX Modify the API of Pipeline and FeatureUnion to match common scikit-learn estimators conventions</a>",378,146,78,0,2022-10-25 10:07:20+00:00
8471,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/8471"">[MRG] Add Gap safe screening to precomputed ElasticNet and MultiTask</a>",1095,364,0,0,2021-04-13 15:34:47+00:00
8474,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/8474"">ENH add support to missing values in NMF</a>",477,93,25,0,2023-04-20 23:03:37+00:00
8547,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/8547"">[MRG] ENH: multi-output support for BaggingRegressor</a>",31,6,19,0,2021-03-02 12:07:40+00:00
8563,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/8563"">[MRG] add random_state in tests estimators</a>",1193,840,11,1,2021-01-22 10:49:28+00:00
8602,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/8602"">FEAT Large Margin Nearest Neighbor implementation</a>",2474,9,515,0,2022-11-24 15:36:08+00:00
8604,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/8604"">[MRG] Make BaseSVC.classes_ sample_weight aware (#8566)</a>",30,5,7,0,2021-01-22 10:49:30+00:00
8663,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/8663"">[MRG] Calibration curve weighted</a>",37,6,26,0,2022-01-16 08:34:38+00:00
8746,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/8746"">Fix F1 F-beta score docstring ambiguities in defining weighted averages</a>",14,3,11,0,2022-07-20 01:39:18+00:00
8848,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/8848"">[WIP] DOC modifying model evaluation documentation</a>",2512,2393,19,0,2021-01-22 10:49:32+00:00
9014,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/9014"">[MRG] partial_fit implementation in TfidfTransformer</a>",63,17,18,0,2023-02-01 15:32:35+00:00
9059,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/9059"">[MRG+1] Added support for sparse multilabel y for Nearest neighbor classifiers</a>",5952,42,89,0,2021-01-22 10:49:35+00:00
9091,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/9091"">[WIP] Eleven point average precision</a>",372,86,16,0,2021-01-22 10:49:36+00:00
9169,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/9169"">[MRG] Dummy 2d fix</a>",20,4,4,0,2021-01-22 10:49:36+00:00
9173,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/9173"">[MRG] Add plotting module with heatmaps for confusion matrix and grid search results</a>",594,66,111,0,2021-01-22 10:49:37+00:00
9179,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/9179"">[MRG+1] Implements SelectDimensionKernel</a>",342,7,31,0,2021-12-13 19:37:04+00:00
9244,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/9244"">[WIP] add RBF kernel to MeanShift package and correct the website description </a>",43,8,4,0,2021-01-22 10:49:38+00:00
9290,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/9290"">[WIP] Adding classes argument to Gaussian Naive Bayes</a>",406,59,177,0,2021-01-22 10:49:38+00:00
9316,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/9316"">[WIP] Chain on decision_function or predict_proba in ClassifierChain</a>",14,3,6,0,2021-12-17 04:15:37+00:00
9334,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/9334"">[MRG] partial_fit for BayesianGaussianMixture</a>",298,18,18,0,2021-12-16 20:47:49+00:00
9413,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/9413"">[MRG] add stratify and shuffle variants for GroupKFold</a>",138,36,16,0,2021-01-22 10:49:42+00:00
9439,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/9439"">[WIP] Make knn kernel undirected.</a>",32,22,4,0,2021-01-22 10:49:43+00:00
9532,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/9532"">[WIP] Add classes parameter to all classifiers</a>",1268,226,25,0,2021-01-22 10:49:44+00:00
9581,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/9581"">[WIP] Gaussian Mixture Model with Missing Data Estimation and Imputation Support</a>",392,4,12,0,2021-01-22 10:49:45+00:00
9585,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/9585"">Add pass classes argument to scorers and allow_subset_labels in metrics</a>",332,43,58,0,2021-07-28 13:37:32+00:00
9615,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/9615"">[WIP] Introducing SoftImpute FactorizationImputer for scikit-learn</a>",170,8,8,0,2021-01-22 10:49:46+00:00
9696,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/9696"">[WIP] FIX avoid making deepcopy in clone</a>",186,136,8,0,2021-01-22 10:49:47+00:00
9741,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/9741"">[MRG] run check_estimator on meta-estimators</a>",98,30,60,0,2022-12-01 14:51:37+00:00
9758,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/9758"">MAINT remove unused stop parameter from compute_gradient</a>",7,10,5,0,2021-04-13 15:34:46+00:00
9822,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/9822"">[MRG] Transposing clustering and classification comparisons</a>",18,21,8,0,2021-01-22 10:49:48+00:00
9834,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/9834"">[MRG] Extending MDS to new data</a>",242,14,44,0,2021-12-23 04:05:50+00:00
9888,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/9888"">[MRG+1] Enforce euclidean metric on LLE</a>",21,1,32,0,2022-02-06 20:51:24+00:00
9894,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/9894"">Adding a \`prefit\` parameter to the VotingClassifier</a>",75,10,12,0,2021-01-22 10:49:50+00:00
10043,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/10043"">[WIP] Voting class logit vote</a>",22,7,5,0,2021-12-22 00:58:47+00:00
10121,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/10121"">[MRG] LoOP: Local Outlier Probabilities</a>",932,112,21,0,2021-03-29 10:47:44+00:00
10163,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/10163"">[WIP] Fix parallel votingClassifier prediction method needed</a>",22,6,8,0,2021-01-22 10:49:56+00:00
10186,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/10186"">Support for masked arrays in MDS</a>",36,15,11,0,2021-01-22 10:49:57+00:00
10266,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/10266"">added sammon's mapping dimensionality reduction in manifold</a>",368,2,8,0,2021-01-22 10:49:58+00:00
10323,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/10323"">[WIP] : Added assert_consistent_docs() and related tests</a>",345,2,81,0,2021-01-22 10:49:59+00:00
10478,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/10478"">[MRG] Fixes #10470: FeatureHasher ordered feature list</a>",209,32,78,0,2021-04-13 15:34:46+00:00
10597,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/10597"">[MRG] Adding LinearKernel to GaussianProcesses</a>",194,22,34,1,2021-01-22 10:50:02+00:00
10604,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/10604"">[MRG]Clusters-Class auto-match</a>",183,3,46,0,2021-11-14 02:15:07+00:00
10612,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/10612"">Added support for not normalising OneVsRestClassifier predict_proba result</a>",58,13,40,1,2021-01-22 10:50:03+00:00
10745,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/10745"">[WIP] Refactor cd</a>",1572,5,9,0,2021-05-20 20:42:06+00:00
10799,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/10799"">[MRG] Calculation of information in nats or bits depending on log_base param</a>",186,51,13,0,2021-04-13 15:34:45+00:00
10875,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/10875"">[MRG+1] Enable using 2SD scaling in StandardScaler</a>",82,12,120,1,2022-01-15 02:14:06+00:00
11046,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/11046"">[MRG] Mean shift: density based cluster assignment</a>",185,67,7,0,2021-01-22 10:50:09+00:00
11054,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/11054"">[MRG] Added stratify option for learning_curve</a>",164,22,44,1,2021-02-25 16:29:08+00:00
11096,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/11096"">[MRG] Implement calibration loss metrics</a>",365,22,145,0,2022-12-31 19:08:16+00:00
11174,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/11174"">[WIP] add max_iter support for partial_fit in SGD and PassiveAgressiveClassifier</a>",69,27,1,0,2021-03-09 09:50:20+00:00
11175,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/11175"">ENH: add oob support to CalibratedClassifier</a>",49,2,7,0,2022-02-14 03:01:56+00:00
11266,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/11266"">[WIP] API: allow cross validation to work with partial_fit</a>",159,18,3,0,2022-02-14 03:03:58+00:00
11282,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/11282"">Fix for #10372: partial_dependence.py _grid_from_X ignores percentile specification for small numbers of unique feature values</a>",15,8,5,0,2021-01-22 10:50:14+00:00
11305,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/11305"">[MRG] Pandas Interoperability section</a>",300,0,76,0,2021-01-22 10:50:15+00:00
11324,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/11324"">[WIP] API specify test parameters via classmethod</a>",360,44,43,0,2021-01-22 10:50:15+00:00
11349,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/11349"">[MRG] FIX allow non-finite target values in TransformedTargetRegressor</a>",29,3,22,1,2021-01-22 10:50:16+00:00
11368,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/11368"">[MRG] New feature: SamplingImputer</a>",491,66,90,1,2022-11-04 15:38:37+00:00
11426,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/11426"">Score Function added</a>",141,0,5,0,2022-01-15 09:23:35+00:00
11430,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/11430"">[MRG] DOC add guideline for choosing a scoring function</a>",161,0,32,0,2023-01-30 15:23:11+00:00
11549,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/11549"">[WIP] use more robust mean online computation in StandardScaler</a>",31,3,10,0,2021-01-22 10:50:20+00:00
11589,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/11589"">Allowing sparse y in GridSearchCV and cross_val_score</a>",217,39,64,2,2021-12-07 04:10:30+00:00
11601,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/11601"">[MRG] Issue 9698 safe and short repr</a>",263,101,27,0,2021-12-21 08:52:02+00:00
11639,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/11639"">[MRG] ENH: Adds inverse_transform to ColumnTransformer</a>",366,13,70,1,2021-01-22 10:50:22+00:00
11649,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/11649"">Fix Tree Median Calculation for MAE criterion</a>",89,36,11,0,2022-01-15 00:14:19+00:00
11671,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/11671"">[MRG] Add Penalty factors for each coefficient in enet ( see #11566)</a>",100,24,9,0,2021-04-13 15:34:43+00:00
11723,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/11723"">[MRG] MLP - add class weights support</a>",166,36,78,0,2023-02-24 19:03:41+00:00
11915,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/11915"">[MRG] Add feature weight to isolation forest</a>",183,6,23,0,2022-01-30 08:56:54+00:00
12192,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/12192"">ENH: Add support for cosine distance in k-means</a>",221,58,4,0,2023-06-01 17:22:31+00:00
12326,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/12326"">[WIP] Extending AdaBoost's base estimator's feature_importances_ attr to support coef_ attr of an estimator</a>",74,34,19,0,2021-03-01 09:53:33+00:00
12364,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/12364"">[WIP] priority of features in decision to splitting</a>",64,31,11,0,2022-02-14 03:15:41+00:00
12457,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/12457"">[MRG] Conditional GMM sampling</a>",126,0,5,0,2021-01-22 10:50:35+00:00
12476,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/12476"">[MRG] Add early exaggeration iterations as argument to t-SNE</a>",33,19,12,0,2021-01-22 10:50:36+00:00
12635,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/12635"">[WIP] pairwise_distances(X) should always have 0 diagonal  intial commit</a>",49,4,17,0,2021-01-22 10:50:38+00:00
12694,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/12694"">Revise plot_out_of_core_classification.py</a>",82,51,31,1,2021-02-25 13:44:40+00:00
12841,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/12841"">[MRG] Implement randomized PCA</a>",522,76,102,0,2021-04-13 15:34:41+00:00
12849,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/12849"">[WIP]  Make SVC tests independent of SV ordering </a>",90,31,5,0,2022-07-16 05:44:06+00:00
12866,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/12866""> NOCATS: Categorical splits for tree-based learners (ctnd.)</a>",1247,239,58,0,2023-07-16 09:53:29+00:00
12898,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/12898"">[MRG] Bugfix for not-yet-confirmed issue #12863: arpack returns singular values in ascending order, the opposite was supposed in sklearn</a>",35,6,15,0,2021-03-25 15:34:36+00:00
12943,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/12943"">ENH Add new smoothing methods to MultinomialNB</a>",112,9,17,0,2021-01-22 10:50:43+00:00
12945,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/12945"">Add a backward compatible flag \`train_score_size\` to the cross_validate function.</a>",24,7,7,0,2021-01-22 10:50:44+00:00
13025,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/13025"">[WIP] Example of multiple imputation with IterativeImputer</a>",394,0,165,0,2022-02-22 04:13:27+00:00
13113,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/13113"">[Feature] Normalized Precision-Recall Metrics Added</a>",299,1,2,0,2022-02-14 03:21:56+00:00
13131,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/13131"">FIX #13117: Remove multiple output from LinearModelCV doc</a>",4,1,17,0,2021-01-22 10:50:48+00:00
13246,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/13246"">[WIP] Common test for equivalence between sparse and dense matrices.</a>",371,64,123,0,2023-08-12 16:11:56+00:00
13269,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/13269"">[WIP] Resamplers</a>",1077,240,93,0,2021-01-22 10:50:51+00:00
13317,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/13317"">[WIP] Change default convergence params for classes using liblinear (LinearSVM and LogisticRegression)</a>",237,22,2,0,2021-01-22 10:50:54+00:00
13351,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/13351"">Pandas DataFrame Categories supported by OneHotEncoder</a>",96,15,1,0,2021-01-22 10:50:56+00:00
13490,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/13490"">Tree export fill colors</a>",60,15,3,0,2022-02-14 03:22:57+00:00
13492,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/13492"">Added vip_ attribute to PLSRegression</a>",16,0,15,0,2023-01-23 07:43:27+00:00
13625,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/13625"">FIX check metric parameters of the given object in neighbors</a>",85,0,19,0,2023-08-08 15:28:44+00:00
13669,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/13669"">Fuzzy c-means clustering algorithm is partly implemented.</a>",243,1,8,0,2022-02-14 03:25:47+00:00
13714,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/13714"">2D sample weight for MultiOutputEstimator</a>",32,8,3,0,2021-01-22 10:51:03+00:00
13761,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/13761"">[MRG] Cross-validation for time series (inserting gaps between the training set and the test set)</a>",672,2,10,0,2021-01-22 10:51:05+00:00
13908,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/13908"">[MRG] Fix formula of the objective function in BayesianRidge</a>",183,84,24,0,2022-02-25 20:25:31+00:00
13935,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/13935"">[MRG] decision_path and apply for all tree-based ensemble model classes.</a>",277,111,6,0,2022-08-17 08:57:05+00:00
13942,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/13942"">[MRG] Adds Bernoulli mixture model to the \`mixture\` module</a>",605,2,8,0,2022-02-01 04:15:40+00:00
13988,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/13988"">Avoid memory copies in linear_model when fit_intercept=False</a>",13,13,6,0,2021-01-22 10:51:08+00:00
14001,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/14001"">Run itrees in parallel during prediction.</a>",51,19,32,1,2022-04-24 01:28:29+00:00
14102,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/14102"">[WIP] Added SelectFromModelCV</a>",154,3,3,0,2021-01-22 10:51:09+00:00
14175,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/14175"">Refactor BaseEstimator's and Kernel's get_params and set_params</a>",134,126,21,0,2022-02-23 03:36:09+00:00
14246,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/14246"">[WIP] FIX make sure sample_weight is taken into account by estimators</a>",38,0,13,0,2021-01-22 10:51:12+00:00
14266,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/14266"">[WIP] FIX make sure sample_weight has an effect on predictions</a>",41,0,3,0,2021-01-22 10:51:12+00:00
14384,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/14384"">Add feature distributed data loading for load_svmlight_file()</a>",138,11,3,0,2021-04-13 15:34:37+00:00
14388,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/14388"">MNT Refactor the three base scorers into _BaseScorer</a>",70,154,3,0,2021-03-10 15:07:04+00:00
14524,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/14524"">[Feature] Adding a Group And Label Kfold Split Method to Model Selection</a>",343,8,21,0,2021-01-22 10:51:16+00:00
14560,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/14560"">[MRG] Binned regression cv</a>",215,18,121,1,2022-05-17 20:25:28+00:00
14636,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/14636"">ENH: Add a variable ""copy"" to perform in-place laplacian calculation in spectral clustering</a>",59,8,10,0,2022-01-05 20:12:09+00:00
14654,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/14654"">[MRG] ENH add support for multiclass-multioutput to ClassifierChain</a>",36,17,26,0,2023-01-11 13:33:47+00:00
14683,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/14683"">[WIP] Added 2 common tests for sparse matrix densification.</a>",141,0,5,0,2021-01-22 10:51:20+00:00
14698,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/14698"">[MRG] Adds Correlation Based Feature Selection</a>",561,30,42,1,2023-02-17 20:52:54+00:00
14748,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/14748"">Added option to use standard idf term for TfidfTransformer and TfidfVectorizer</a>",43,18,37,1,2022-09-13 16:57:19+00:00
14859,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/14859"">[WIP] - Precision and recall at k</a>",318,0,5,0,2022-05-02 01:14:52+00:00
14942,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/14942"">Warning the user of bad default values, starting by dbscan.eps</a>",105,4,25,0,2022-08-08 08:54:34+00:00
14963,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/14963"">Pass through non-ndarray object that support nep13 and nep18: vaex+sklearn out of core</a>",7,0,9,0,2021-01-22 10:51:26+00:00
14966,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/14966"">Deprecate TfidfVectorizer</a>",152,65,11,0,2022-02-14 05:04:40+00:00
14984,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/14984"">[MRG] Sorting ordering option in OrdinalEncoder</a>",75,18,48,0,2022-03-23 21:32:28+00:00
15032,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/15032"">[WIP] Fixing data leak with warm starting in GBDT</a>",65,6,12,0,2021-01-22 10:51:28+00:00
15050,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/15050"">[MRG] ENH Adds warning with pandas category does not match lexicon ordering</a>",133,1,47,2,2022-04-07 13:34:04+00:00
15136,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/15136"">[WIP]  Refactoring logic for chosing eigen solver in spectral clustering (#14713)</a>",74,45,1,0,2022-05-07 23:33:19+00:00
15176,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/15176"">[WIP] Implement Gini coefficient for model selection with positive regression GLMs</a>",3012,59,2,0,2021-01-22 10:51:31+00:00
15403,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/15403"">ENH fast path for binary confusion matrix</a>",36,24,11,0,2022-01-15 11:52:13+00:00
15504,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/15504"">[MRG] BUG: enforce row-wise arg max of decision function=predicted class</a>",24,3,5,0,2022-03-10 03:54:46+00:00
15526,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/15526"">ENH: implement __sizeof__ for estimators</a>",61,0,17,0,2021-12-16 20:36:36+00:00
15583,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/15583"">MRG Logistic regression preconditioning</a>",344,60,109,0,2023-01-24 17:05:45+00:00
15648,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/15648"">[MRG] FIX and ENH in _RidgeGCV</a>",160,89,16,0,2021-01-22 10:51:41+00:00
15702,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/15702"">[MRG] ENH Improves on the plotting api by storing weak ref to display in axes</a>",207,42,16,0,2021-01-22 10:51:43+00:00
15710,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/15710"">[WIP] Add support for Histogram-Like Multi-Labeled Encodings with Associated Values</a>",251,0,4,0,2021-01-22 10:51:43+00:00
15737,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/15737"">[MRG] metrics.most_confused_classes (issue #15696)</a>",174,3,18,0,2021-01-22 10:51:43+00:00
15815,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/15815"">Mod: Modify silhouette_score(), calinski_harabasz_score() and davies_…</a>",46,48,10,0,2022-03-29 03:28:50+00:00
15854,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/15854"">[MRG] FIX provide predictions in the original space in RidgeCV</a>",49,4,6,0,2021-01-22 10:51:45+00:00
15856,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/15856"">test_svm_equivalent_sample_weight_C should use tighter tolerance</a>",2,2,9,0,2022-08-02 16:15:54+00:00
15922,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/15922"">[WIP] Add \`sparse-rbf\` kernel option for \`semi_supervised.LabelSpreading\`</a>",474,24,1,0,2021-01-22 10:51:46+00:00
15931,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/15931"">[WIP] cd_fast speedup</a>",5,9,9,0,2022-03-23 10:05:49+00:00
15982,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/15982"">Affinity propagation preference percentile</a>",36,7,2,0,2021-01-22 10:51:49+00:00
15985,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/15985"">Filter empty samples after removing metadata from 20 newsgroup dataset</a>",28,7,3,0,2021-01-22 10:51:49+00:00
16010,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/16010"">[WIP] Kernel cookbook -- additional kernels for Gaussian Process Regression</a>",1326,1396,6,0,2021-01-22 10:51:50+00:00
16029,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/16029"">[WIP] BUG returned most regularized model with Ridge*CV in case of tie</a>",32,9,2,0,2022-05-19 16:23:31+00:00
16185,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/16185"">[MRG] ENH avoid deepcopy when a parameter is declared immutable</a>",176,39,9,0,2022-10-05 07:11:11+00:00
16194,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/16194"">Meanshift alternate implementation issue16171</a>",205,0,2,0,2021-01-22 10:51:55+00:00
16196,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/16196"">Use np.empty_like in PolynomialFeatures for NEP18 support</a>",130,4,19,0,2021-01-22 10:51:56+00:00
16236,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/16236"">FEA Group aware Time-based cross validation</a>",1228,9,153,0,2023-07-20 15:21:44+00:00
16369,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/16369"">ENH: Conditional Density Estimation Loss - New Decision Tree Criterion</a>",151,12,3,0,2021-04-13 15:34:33+00:00
16418,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/16418"">[WIP] Example generating various distance metrics on RandomTreesEmbedding and </a>",187,0,0,0,2022-08-24 21:04:19+00:00
16439,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/16439"">[WIP] PERF Parallelize W/H updates of NMF with OpenMP</a>",27,15,6,0,2023-07-23 19:38:57+00:00
16463,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/16463"">added LinearOperator support in safe_sparse_dot()</a>",66,6,37,1,2021-04-10 16:58:45+00:00
16574,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/16574"">WIP Enabling different array types (CuPy) in PCA with NEP 37</a>",191,128,32,0,2021-01-22 10:52:09+00:00
16581,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/16581"">Implement Extended Isolation Forest</a>",183,3,8,0,2021-11-26 23:06:43+00:00
16650,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/16650"">Improved calibration example plots</a>",203,104,12,0,2021-01-22 10:52:11+00:00
16714,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/16714"">[MRG] Allow only fit_transform to be present in pipeline</a>",81,10,57,1,2022-01-14 13:48:38+00:00
16721,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/16721"">Improve IsolationForest average depth evaluation</a>",9,42,19,0,2021-01-22 10:52:13+00:00
16800,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/16800"">Add sparse support for 'y' to multioutput regression</a>",44,1,13,0,2022-01-23 06:38:24+00:00
16834,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/16834"">[MRG] FEA Gower distance</a>",584,46,55,0,2021-08-13 04:30:54+00:00
16858,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/16858"">Nonlinear regression simulations for split criteria comparison</a>",756,0,4,0,2021-11-05 21:22:49+00:00
16881,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/16881"">[MRG] OneHotEncoder handle_unknown param can be set to 'warning'</a>",65,30,0,0,2022-08-24 21:06:33+00:00
16898,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/16898"">Tutorial on warmstart grid search for multiple classifiers</a>",291,0,3,0,2022-01-04 02:33:07+00:00
16994,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/16994"">[MRG] ENH Support column vector input to feature extraction</a>",156,14,9,0,2021-01-22 10:52:24+00:00
17048,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/17048"">New Feature: return_all_estimators option in GridSearch and RandomizedSearch</a>",54,15,7,0,2021-01-22 10:52:26+00:00
17052,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/17052"">ENH Adds permute_y to make_classification interaction</a>",32,1,3,0,2022-08-02 21:09:57+00:00
17130,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/17130"">Gaussian Mixture - weighted implementation </a>",619,345,15,0,2022-04-12 12:19:20+00:00
17137,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/17137"">Add sample table for Error-Correcting Output-Codes</a>",24,9,1,0,2021-01-22 10:52:27+00:00
17141,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/17141"">[MRG] DOC Fix KFold and GroupKFold</a>",2,2,5,0,2022-08-09 05:06:53+00:00
17179,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/17179"">[WIP] FIX KBinsDiscretizer: allow nans #9341</a>",52,8,17,0,2021-12-17 04:19:47+00:00
17377,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/17377"">[WIP] BUG make _weighted_percentile behave as NumPy</a>",434,164,40,0,2021-01-22 10:52:35+00:00
17378,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/17378"">[WIP] Add information gain ratio criterion to tree classifiers</a>",99,11,4,0,2021-04-13 15:34:30+00:00
17441,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/17441"">Run common checks on estimators with non default parameters</a>",311,0,15,0,2021-01-22 10:52:39+00:00
17541,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/17541"">[MRG] Add class_weight parameter to CalibratedClassifierCV</a>",123,27,68,2,2022-05-10 22:08:54+00:00
17544,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/17544"">[WIP] Use dataset factories for estimator checks</a>",38,29,5,0,2021-12-16 21:45:31+00:00
17575,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/17575"">escape strings for export_graphviz</a>",48,0,6,0,2021-01-22 10:52:42+00:00
17642,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/17642"">Tutorial on how to create a custom kernel for Gaussian Process</a>",94,1,27,0,2021-10-06 09:52:54+00:00
17676,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/17676"">[WIP] PCA NEP-37 adding random pathway and CuPy test</a>",346,157,17,0,2021-01-22 10:52:44+00:00
17711,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/17711"">Implement probability estimates for NearestCentroid classifier</a>",70,4,4,0,2023-06-21 23:04:00+00:00
17744,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/17744"">[WIP] NEP-18 support for preprocessing algorithms</a>",204,12,15,0,2022-10-10 18:07:28+00:00
17768,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/17768"">ENH improve _weighted_percentile to provide several interpolation</a>",521,60,45,0,2022-07-05 09:46:02+00:00
17775,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/17775"">ENH add interpolation parameter to DummyRegressor for ""median"" and ""quantile"" strategies</a>",431,71,0,0,2021-01-22 10:52:49+00:00
17799,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/17799"">ENH Adds typing support to LogisticRegression</a>",26,15,39,0,2022-06-30 12:19:29+00:00
17806,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/17806"">MNT Deprecates _estimator_type and replaces by a estimator tag</a>",280,44,68,0,2021-01-22 10:52:51+00:00
17843,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/17843"">SubsampledNeighborsTransformer: Subsampled nearest neighbors for faster and more space efficient estimators that accept precomputed distance matrices</a>",608,2,22,0,2022-10-23 00:49:27+00:00
17872,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/17872"">WIP option to cluster classes in confusion matrix</a>",79,1,11,0,2021-01-22 10:52:53+00:00
17889,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/17889"">[WIP] ENH create a generator of applicable metrics depending on the target y</a>",578,166,3,0,2021-01-22 10:52:54+00:00
17929,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/17929"">DOC Pilot of annotating parameters with category</a>",62,8,26,0,2021-06-01 02:45:28+00:00
17930,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/17930"">[WIP] ENH create callable class to get adequate scorer for a problem</a>",337,4,4,0,2021-01-22 10:52:55+00:00
17949,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/17949"">[WIP] Allow TruncatedSVD using randomized to automatically reset k < n_features.</a>",5,2,9,0,2022-01-10 20:59:49+00:00
17962,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/17962"">EHN allow scorers to set addtional parameter of scoring function</a>",230,33,28,0,2021-04-13 16:08:54+00:00
17975,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/17975"">Remove single-target assertion in LinearModelCV</a>",28,28,5,0,2021-01-22 10:52:56+00:00
17991,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/17991"">allowing string input for pairwise_distances</a>",59,9,16,0,2022-11-03 09:11:03+00:00
17999,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/17999"">[WIP] POC for Factory-style construction of composite estimators</a>",113,72,0,0,2021-01-22 10:52:57+00:00
18118,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/18118"">Rewrite fast_mcd to be more memory-efficient</a>",146,246,1,0,2021-11-19 22:18:15+00:00
18141,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/18141"">ENH allow to pass str or scorer to make_scorer</a>",111,13,11,0,2022-04-24 13:50:31+00:00
18155,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/18155"">[MRG] Fix Single linkage option in Agglomerative causes MemoryError for very large numbers</a>",20,8,6,0,2021-11-17 21:18:12+00:00
18292,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/18292"">[MRG] Test __array_function__ not called in non-estimator API  (#15865)</a>",25,0,10,0,2022-01-24 15:46:54+00:00
18390,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/18390"">[MRG] Added option to return raw predictions from cross_validate</a>",82,5,13,0,2022-03-01 16:53:52+00:00
18413,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/18413"">[FEAT] Weighting option in MDS</a>",244686,114241,8,0,2023-07-30 15:09:44+00:00
18422,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/18422"">[MRG] FEA Add MDLP discretization</a>",1371,246,3,0,2022-12-13 20:39:44+00:00
18479,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/18479"">FEA Add cumulative gain curve metric</a>",3743,81,9,0,2021-12-28 12:32:25+00:00
18492,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/18492"">[WIP] online matrix factorization with missing values</a>",280,19,0,0,2021-01-22 10:53:15+00:00
18521,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/18521"">[MRG] ComplementNB with class priors when computing joint log-likelihoods (Trac : #14523, #14444)</a>",3,2,1,0,2022-03-23 02:43:58+00:00
18555,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/18555"">API Deprecate using labels in bytes format</a>",59,12,78,3,2023-08-08 02:58:26+00:00
18603,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/18603"">ENH: OOB Permutation Importance for Random Forests</a>",957,224,132,1,2022-04-14 18:57:37+00:00
18632,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/18632"">[MRG] Fix svm gui readability</a>",18,1,2,0,2021-01-22 10:53:20+00:00
18679,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/18679"">[WIP] 'most_frequent' drop method for OneHotEncoder</a>",155,11,6,0,2023-04-05 12:33:28+00:00
18689,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/18689"">[WIP] Allow fitting PCA on sparse X with arpack solvers</a>",106,29,19,0,2023-05-24 22:31:52+00:00
18738,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/18738"">ENH Extend plot_precision_recall_curve and plot_roc_curve to multiclass</a>",516,135,4,0,2021-01-22 10:53:25+00:00
18750,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/18750"">ENH allows checks generator to be pluggable</a>",116,14,25,0,2021-01-22 10:53:27+00:00
18786,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/18786"">[WIP] Add alpha parameter to GaussianProcessClassifier</a>",44,4,5,0,2021-02-06 18:32:47+00:00
18821,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/18821"">DOC add example to show how to deal with cross-validation</a>",229,0,109,1,2023-07-27 12:41:46+00:00
18843,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/18843"">ENH ensure no copy if not requested and improve transform performance in TFIDFTransformer</a>",16,12,18,0,2021-04-25 22:18:33+00:00
18860,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/18860"">[MRG] SimpleImputer: Handle string features where all values are missing</a>",32,1,3,0,2021-01-22 10:53:35+00:00
18889,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/18889"">[MRG] Add partial_fit function to DecisionTreeClassifier</a>",676,22,15,0,2021-12-02 14:14:28+00:00
18942,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/18942"">FIX Improved the error message from check_classification_targets function when label is invalid</a>",6,2,0,0,2022-08-24 21:09:47+00:00
18953,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/18953"">Improve dense SVC performance by reimplementing dense SVC rbf kernel with GEMV BLAS API</a>",273,38,4,0,2021-11-02 17:01:32+00:00
18970,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/18970"">Implement binary/multiclass classification metric - Spherical Payoff</a>",68,0,2,0,2021-01-22 10:53:42+00:00
19043,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19043"">added ValueError for improperly shaped Y when doing a multilabel fit</a>",33,0,0,0,2022-08-24 21:10:51+00:00
19065,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19065"">Add resample to AdaBoostClassifier in _weight_boosting.py</a>",36,3,3,0,2021-01-22 10:53:46+00:00
19079,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19079"">[MRG] Correct handling of missing_values and NaN in SimpleImputer for object arrays (closes #19071)</a>",78,7,2,0,2022-12-24 04:27:36+00:00
19087,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19087"">FIX Isolation forest path length for small samples</a>",125,14,61,0,2022-12-24 04:29:30+00:00
19181,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19181"">Create plot_contigency_matrix.py</a>",107,0,1,0,2021-01-22 10:53:53+00:00
19187,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19187"">[WIP] Add sparse matrix support for histgradientboostingclassifier</a>",190,13,17,0,2022-03-28 00:58:37+00:00
19190,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19190"">Enhancement to Confusion Matrix Output Representation for improving readability #19012 </a>",73,12,6,0,2022-11-06 09:32:40+00:00
19201,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19201"">kernel pca doc additions</a>",67,3,1,0,2022-09-20 01:33:03+00:00
19223,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19223"">[MRG] Support for multiouput in VotingRegressor and VotingClassifier (Fixes #18289)</a>",233,42,1,0,2022-02-07 10:37:50+00:00
19252,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19252"">Document differences to the BIRCH algorithm</a>",38,6,21,0,2023-01-04 16:54:08+00:00
19253,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19253"">TST Add test for numerical issues in BIRCH</a>",35,1,10,0,2021-08-04 21:36:24+00:00
19254,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19254"">[MRG] Feature Proximity Matrix in RandomForest class</a>",38,0,2,0,2021-01-24 11:16:48+00:00
19392,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19392"">[MRG] Improved parallel execution of plot_partial_dependence / partial_dependence</a>",115,44,7,0,2021-02-20 10:35:22+00:00
19427,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19427"">Fixed bug handling multi-class classification</a>",33,8,11,0,2022-07-09 03:26:52+00:00
19498,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19498"">[MRG] Fix Improvement in RANSAC: is_data_valid should receive subset_idxs </a>",14,7,9,0,2021-08-02 12:15:55+00:00
19526,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19526"">BIC and AIC scores in _bayesian_mixture.py</a>",85,0,0,0,2022-08-24 21:14:37+00:00
19556,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19556"">FEA Confusion matrix derived metrics</a>",1253,15,122,0,2023-07-19 12:03:16+00:00
19562,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19562"">[MRG] GaussianMixture with BIC/AIC</a>",1119,17,136,0,2022-05-09 20:57:34+00:00
19589,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19589"">Updated notes in documentation regarding macro-F1 in _classification.py  </a>",8,0,4,0,2021-03-11 14:38:43+00:00
19630,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19630"">DEP start to raise warning with inconsistent combination of hyperparameters in SVM</a>",21,0,4,0,2021-07-29 13:39:02+00:00
19644,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19644"">Implemented persistent random seeds for OOB score calculation in random forests</a>",19,47,0,0,2022-08-24 21:17:14+00:00
19707,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19707"">ENH BestSplitter split on feature with lowest index when multiple features have identical order</a>",168,9,14,0,2022-12-01 14:54:31+00:00
19719,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19719"">skip check_classifiers_predictions test for poor_scoor estimators</a>",3,4,2,0,2021-03-21 17:34:48+00:00
19731,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19731"">ENH Improve regularization messages for QuadraticDiscriminantAnalysis</a>",44,29,24,0,2021-07-27 09:24:13+00:00
19746,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19746"">FIX Ridge.coef_ return a single dimension array when target type is not continuous-multiple</a>",7,6,3,0,2022-04-22 13:59:16+00:00
19751,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19751"">[MRG] Add min_tpr parameter to roc_auc_score, allow both min_tpr and max_fpr </a>",392,45,21,0,2023-07-06 10:56:52+00:00
19753,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19753"">FIX VotingClassifier handles class weights</a>",68,1,21,0,2022-04-28 13:10:26+00:00
19754,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19754"">Draft implementation of non-parametric quantile methods (RF, Extra Trees and Nearest Neighbors)</a>",1791,178,35,0,2023-01-26 17:21:51+00:00
19789,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19789"">DOC Add warnings about causal interpretation of coefficients</a>",38,0,11,0,2021-06-06 20:40:08+00:00
19854,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19854"">[WIP] Adding Local Regression (Fixes #3075)</a>",406,0,12,0,2023-08-12 14:48:49+00:00
19874,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19874"">ENH add parameter in accuracy_score for multilabel classification</a>",141,28,1,0,2021-04-13 09:33:37+00:00
19914,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19914"">WIP Adds Generalized Additive Models with Bagged Hist Gradient Boosting Trees</a>",669,53,14,0,2021-10-15 18:44:40+00:00
19928,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19928"">WIP KBinsDiscretizer supports NaN as input values</a>",28,5,0,0,2022-08-01 18:05:06+00:00
19938,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19938"">TST add common test to check support for bytes target</a>",50,0,0,0,2021-08-31 09:39:53+00:00
19958,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19958"">DOC Added references to the Matthews correlation coefficient function in the user guide</a>",28,0,4,0,2022-02-14 00:31:12+00:00
19971,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19971"">EHN Add transform_inverse to Nystroem</a>",116,5,2,0,2021-04-27 08:58:34+00:00
19973,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/19973"">Fix Common tests xfailed for Bicluster estimators issue #19548</a>",14,16,0,0,2021-04-25 19:52:50+00:00
20003,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/20003"">DOC add cross-reference to examples instead of duplicating content for GPR</a>",48,174,46,0,2023-08-11 22:37:16+00:00
20029,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/20029"">Graph Spectral Embedding to sklearn/manifold</a>",1029,0,31,0,2023-01-12 14:52:37+00:00
20043,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/20043"">[affinity propagation] change logic in edge case where all similarities and preferences are equal</a>",1,1,5,0,2023-08-02 16:02:44+00:00
20070,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/20070"">FIX use correct formulation for ExpSineSquared kernel</a>",111,45,4,0,2023-06-08 11:28:04+00:00
20144,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/20144"">TST suppress convergence warnings in coordinate descent</a>",76,16,36,0,2022-09-11 09:03:11+00:00
20233,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/20233"">[MRG] FIX ""Add the ability for the user to provide specific landmarks points to approximate the Nystroem kernel approximation ""</a>",80,8,8,0,2022-10-07 06:59:14+00:00
20287,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/20287"">FIX ColumnTransformer raise TypeError when remainder columns have incompatible dtype</a>",7,1,2,0,2022-07-02 07:07:56+00:00
20494,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/20494"">Fix bug default marking of unlabeled data</a>",40,4,3,0,2021-12-23 00:44:47+00:00
20568,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/20568"">Initialize n_samples in classification criteria</a>",7,6,2,0,2021-07-21 20:03:36+00:00
20603,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/20603"">TST common test for predictions shape consistency with single target</a>",36,0,10,0,2021-11-30 13:39:10+00:00
20604,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/20604"">TST add common tests for linear models for shape consistency of coef</a>",106,8,0,0,2021-07-26 09:36:56+00:00
20608,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/20608"">TST split API checks from other checks</a>",222,132,3,0,2021-08-22 10:48:32+00:00
20825,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/20825"">add new _search_successive_halving.py</a>",979,1046,4,0,2022-04-14 18:03:33+00:00
20856,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/20856"">[WIP] SequentialFeatureSelector stores the intermediate scores and the history of the support masks</a>",20,7,1,0,2021-08-27 11:07:25+00:00
21010,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/21010"">Adding a reset weights strategy to Adaboost Regressor</a>",111,10,10,0,2022-09-14 18:01:55+00:00
21033,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/21033"">Add custom_range argument for partial dependence</a>",426,53,47,0,2023-04-17 18:34:49+00:00
21058,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/21058"">ENH HTML repr shows best estimator in *SearchCV when refit=True</a>",63,1,4,0,2021-10-09 12:57:22+00:00
21132,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/21132"">ENH add new common test: \`test_negative_sample_weight_support\` and \`allow_negative_sample_weight\` tag </a>",163,9,4,0,2022-10-05 08:16:51+00:00
21143,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/21143"">FIX \`CalibratedClassifierCV\` should not ignore \`sample_weight\` if estimator does not support it</a>",95,28,7,0,2023-06-08 09:49:47+00:00
21211,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/21211"">ENH use cv_results in the different curve display to add confidence intervals</a>",465,40,7,0,2021-11-19 18:26:48+00:00
21225,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/21225"">Added support for sampling for exponential, linear and epanechnikov kernels in KernelDensity</a>",104,43,6,0,2022-09-14 04:10:06+00:00
21292,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/21292"">Add partial model evaluation to gradient boosting</a>",56,3,3,0,2022-02-17 15:16:13+00:00
21311,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/21311"">Fix column_transformer to use fitparams like Pipeline</a>",93,5,8,0,2021-10-28 18:07:12+00:00
21320,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/21320"">[MRG] FEA Lift metric and curve</a>",1104,0,5,0,2021-12-22 01:17:57+00:00
21354,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/21354"">[MRG] Add alt text to scikit-learn documentation</a>",63,2,8,0,2021-11-28 21:10:20+00:00
21409,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/21409"">[feat] enable the n_jobs for mutual info regression and classifier</a>",121,13,1,0,2021-10-23 11:45:12+00:00
21458,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/21458"">ENH Added vocabulary_count_ attribute to CountVectorizer </a>",85,0,7,0,2022-07-25 22:43:28+00:00
21571,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/21571"">add graph-based clustering</a>",407,0,1,0,2022-09-14 22:43:20+00:00
21691,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/21691"">FET add retry mechanism for fetch_xx functions</a>",253,20,37,2,2022-12-28 17:54:50+00:00
21784,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/21784"">Initial implementation of GridFactory</a>",183,5,12,0,2021-11-29 21:53:54+00:00
21801,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/21801"">[WIP] CountVectorizer and TfidfVectorizer option to featurize out-of-vocab.</a>",453,49,0,0,2021-11-29 05:53:38+00:00
21807,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/21807"">FIX Corrects tags for pipeline and RFE</a>",89,9,0,0,2021-11-28 04:16:14+00:00
21811,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/21811"">FIX Sets feature_names_in_ for estimators in Bagging*</a>",154,82,24,1,2022-04-07 19:19:46+00:00
21839,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/21839"">FIX make sure to limit ClassifierChain to multilabel-indicator target</a>",30,6,0,0,2021-11-30 18:32:02+00:00
21841,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/21841"">TST make sure to have a real multioutput target in common tests</a>",8,3,3,0,2021-12-18 22:06:53+00:00
21935,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/21935"">ENH add x and y to importance getter rfe</a>",66,12,25,0,2023-06-12 09:44:11+00:00
21942,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/21942"">Extend ClassifierChain to multi-output problems (ClassifierChain.decision_function)</a>",10,5,7,0,2022-01-16 08:27:33+00:00
21960,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/21960"">[WIP] decision_function for MultiOutputClassifier</a>",115,1,10,0,2022-09-15 00:13:42+00:00
21962,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/21962"">Added a new method to specify monotonicity constraints using dict of feature names in HistGradientBoostingClassifier/Regressor. Fixes #21961</a>",63,9,2,0,2022-09-15 19:45:42+00:00
22000,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/22000"">[WIP] Callback API continued</a>",1969,22,15,0,2022-10-13 14:27:45+00:00
22004,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/22004"">ENH column_or_1d cast pandas boolean extension to bool ndarray</a>",37,3,0,0,2022-08-01 18:40:17+00:00
22008,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/22008"">fix skip unnecessary check of gram_matrix in coordinate_descent</a>",5,1,5,0,2023-03-23 02:59:28+00:00
22010,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/22010"">[WIP] Make it possible to pass an arbitrary classifier as method for CalibratedClassifierCV</a>",114,22,10,0,2023-01-11 12:57:50+00:00
22021,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/22021"">FIX Add BaggingClassifier support for class_weights with string labels</a>",44,1,0,0,2021-12-21 15:39:57+00:00
22022,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/22022"">[MRG] Add a separator between a criteria and other contents in a node in export_graphviz for visibility</a>",95,21,1,0,2022-04-24 13:54:30+00:00
22039,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/22039"">WIP Ensemble Classifiers handles class weights</a>",50,6,1,0,2021-12-21 15:22:09+00:00
22043,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/22043"">FEA add pinball loss to SGDRegressor</a>",288,10,47,0,2023-01-11 13:07:23+00:00
22046,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/22046"">ENH Add Multiclass Brier Score Loss</a>",431,76,15,0,2023-03-26 14:16:53+00:00
22078,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/22078"">first attempt to change iterative_imputer</a>",68,38,3,0,2022-10-30 05:10:33+00:00
22227,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/22227"">[MRG] [ENH] Adds a \`return_std_of_f\` flag for GPCs</a>",150,8,1,0,2022-12-21 21:21:28+00:00
22233,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/22233"">[WIP] Brier score binless decomposition</a>",101,0,4,0,2022-06-27 14:41:25+00:00
22285,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/22285"">ENH Support 2d y_score with 2 classes in top_k_accuracy_score w/ labels</a>",13,1,0,0,2022-01-24 17:01:50+00:00
22330,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/22330"">ENH Add eigh as a solver in MDS</a>",241,34,20,0,2023-07-27 14:08:00+00:00
22417,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/22417"">Kernel Ridge Classifier</a>",352,5,1,0,2022-08-03 20:08:07+00:00
22431,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/22431"">Correct precision at k</a>",620,2,10,0,2022-08-24 16:18:56+00:00
22485,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/22485"">[MRG] separate penalty factors for GLM regressions</a>",93,12,34,0,2023-07-08 20:46:42+00:00
22561,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/22561"">Modifications to LinearRegression documentation.</a>",6,4,11,0,2022-12-08 18:16:33+00:00
22573,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/22573"">ENH: Add refit callable generator for model sub-selection with Grid/RandomSearchCV</a>",1323,108,29,0,2023-07-13 15:40:30+00:00
22574,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/22574"">ENH add Naive Bayes Metaestimator \`ColumnwiseNB\` (aka ""GeneralNB"")</a>",1625,6,148,2,2023-07-20 12:49:23+00:00
22606,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/22606"">ENH Uses __sklearn_tags__ for tags instead of mro walking</a>",552,344,2,0,2022-10-26 15:55:55+00:00
22705,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/22705"">ENH Paired haversine</a>",93,2,87,1,2023-08-01 10:16:07+00:00
22733,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/22733"">[MRG] Set the target of datasets to be pandas categoricals</a>",14,4,0,0,2022-05-10 20:12:37+00:00
22754,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/22754"">FEA Add Oblique trees and oblique splitters to tree module: Enables extensibility of the trees</a>",2535,155,23,0,2023-01-30 20:32:55+00:00
22923,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/22923"">ENH Support cardinality filtering of columns in \`make_column_selector\`</a>",67,4,8,0,2022-03-28 13:06:52+00:00
22999,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/22999"">ENH alternative tighter dual gap in elastic net</a>",78,11,2,0,2022-04-02 12:50:32+00:00
23031,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/23031"">ENH Added warning for RidgeCV</a>",47,1,8,0,2022-05-25 13:44:35+00:00
23045,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/23045"">Add sample_weight to the calculation of alphas in enet_path and LinearModelCV</a>",53,37,27,0,2023-07-06 10:56:43+00:00
23111,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/23111"">Fix RegressionChain does not accept nans, when base_estimator does #23109</a>",87,11,10,0,2022-05-11 10:05:51+00:00
23153,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/23153"">[FEAT] Implement quantile SVR</a>",692,38,18,0,2023-03-18 16:45:56+00:00
23183,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/23183"">ENH add zero_division=nan for classification metrics</a>",316,101,71,0,2023-02-02 13:41:13+00:00
23224,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/23224"">FIX ""CategoricalNB fails with sparse matrix"" #16561</a>",53,17,19,0,2022-05-04 07:47:43+00:00
23286,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/23286"">FET add support for manhattan distances in KNN imputer</a>",243,39,9,0,2022-08-01 15:19:03+00:00
23317,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/23317"">MNT: Make error message clearer for n_neighbors</a>",43,2,31,3,2022-12-23 18:25:03+00:00
23346,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/23346"">FIX Enable SelfTrainingClassifier to work with vectorizers</a>",87,40,2,0,2022-05-12 20:47:37+00:00
23371,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/23371"">FIX apply sample weight to RANSAC residual threshold</a>",36,15,5,0,2022-06-24 01:38:23+00:00
23386,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/23386"">FET support feature selection based on permutation importance</a>",191,27,2,0,2022-08-30 18:52:05+00:00
23551,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/23551"">DOC Improvements to developer documentation</a>",9,0,14,0,2023-02-19 19:11:59+00:00
23589,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/23589"">DOC Replace \`chi2\` with \`f_classif\` in feature selection examples</a>",24,24,0,0,2022-07-16 02:00:04+00:00
23603,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/23603"">[WIP] Enable multi-output voting regression</a>",37,16,1,0,2022-06-15 12:07:36+00:00
23616,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/23616"">WIP ENH Added \`auto\` option to \`FastICA.whiten_solver\`</a>",157,29,12,1,2023-07-27 13:54:17+00:00
23644,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/23644"">FIX Adopted more direct \`dtype\` restraint in preperation for \`NEP50\`</a>",5,5,2,0,2023-03-08 16:23:23+00:00
23746,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/23746"">[WIP] DOC Explain missing value mechanisms</a>",27,2,10,0,2023-07-07 15:56:46+00:00
23748,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/23748"">[WIP] Enhance calibration plots</a>",434,84,46,0,2023-02-19 19:09:17+00:00
23780,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/23780"">ENH Add option \`n_splits='walk_forward'\` in \`TimeSeriesSplit\`</a>",158,19,29,1,2023-06-15 10:11:31+00:00
23824,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/23824"">FEA Add strategy isotonic to calibration curve</a>",65,21,6,0,2022-09-06 09:12:52+00:00
23876,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/23876"">[MRG] PERF Significant performance improvements in Partial Least Squares (PLS) Regression</a>",289,136,55,0,2023-01-21 17:05:54+00:00
23931,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/23931"">[WIP | First Contribution] add first pass at TimeSeriesInitialSplit</a>",109,1,8,0,2022-07-21 13:22:40+00:00
23962,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/23962"">[WIP] Make random_state accept np.random.Generator</a>",383,9,17,0,2022-07-29 10:57:57+00:00
24034,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/24034"">Add a page on the website to provide directions to users on how to reach out</a>",34,8,17,0,2023-01-18 08:54:49+00:00
24053,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/24053"">EHN: RadiusNeighborRegressor speedup</a>",62,19,6,0,2023-01-12 17:03:35+00:00
24079,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/24079"">Add \`sample_weight\` support to PCA</a>",197,23,2,0,2022-08-05 00:59:45+00:00
24099,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/24099"">FEA Add \`ArccosDistance\`</a>",29,2,1,0,2022-08-12 13:23:36+00:00
24106,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/24106"">FEA Add \`ArccosDistance\` (continued)</a>",105,7,21,0,2022-11-03 09:42:09+00:00
24119,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/24119"">FIX Correct \`GaussianMixture.weights_\` normalization</a>",1,1,21,0,2023-07-27 15:40:14+00:00
24121,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/24121"">Adds gain scoring metrics</a>",1287,0,11,0,2023-05-31 11:20:27+00:00
24188,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/24188"">FEA Add sample variance support for \`GaussianProcessRegressor.fit\`</a>",112,30,13,0,2023-02-08 09:05:25+00:00
24216,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/24216"">[WIP] example: showcase advantage of balanced_accuracy_score for imbalanced data</a>",120,0,0,0,2022-08-20 18:19:26+00:00
24227,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/24227"">Add RepeatedStratifiedGroupKFold</a>",130,3,0,0,2022-08-23 12:26:48+00:00
24239,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/24239"">[WIP] ENH Tree Splitter: 50% performance improvement with radix sort and feature ranks</a>",189,8,6,0,2023-02-19 18:44:02+00:00
24321,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/24321"">ENH add dtype preservation to FactorAnalysis</a>",98,9,3,1,2023-02-28 18:01:07+00:00
24337,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/24337"">ENH Add dtype preservation to LocallyLinearEmbedding</a>",123,23,10,0,2022-11-03 15:56:35+00:00
24346,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/24346"">ENH Add dtype preservation to FeatureAgglomeration</a>",29,5,9,0,2022-11-07 06:41:40+00:00
24415,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/24415"">[WIP] Implement PCA on sparse noncentered data</a>",369,36,12,0,2023-05-24 16:57:32+00:00
24616,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/24616"">TST use global_random_seed in sklearn/mixture/tests/test_gaussian_mixture.py</a>",126,116,15,0,2023-03-18 23:34:10+00:00
24674,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/24674"">FEA Allow string input for pairwise distances</a>",109,12,40,0,2023-01-11 09:49:23+00:00
24678,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/24678"">MAINT Introduce \`BaseCriterion\` as a base abstraction for \`Criterion\`s</a>",94,59,65,0,2023-07-29 09:17:25+00:00
24681,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/24681"">EHN accept l2 metric with ward linkage in agglomeration clustering</a>",50,17,2,0,2022-10-17 16:54:21+00:00
24746,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/24746"">[Refactor, Tree] Python tree class for modularity and consistency of BaseEstimator</a>",118,96,0,0,2022-11-03 15:47:34+00:00
24751,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/24751"">FEA Ensemble selection from Librairies of Models</a>",1250,0,4,0,2023-02-05 14:58:03+00:00
24788,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/24788"">[WIP] first step in fixing minimum 2 required samples for fixing MLPRegressor attribute error</a>",19,0,18,0,2022-12-01 18:31:41+00:00
24802,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/24802"">TST use global_random_seed in sklearn/cluster/tests/test_spectral.py</a>",39,24,2,1,2022-12-26 20:41:47+00:00
24810,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/24810"">[MRG] Add probabilistic estimates to HistGradientBoostingRegressor</a>",4361,419,3,0,2022-11-24 21:59:35+00:00
24812,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/24812"">ENH Make GMM initialization more robust and easily user customizeable</a>",408,23,8,0,2023-01-21 05:10:27+00:00
24834,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/24834"">Specify the constraints for \`metric=""precomputed""\`</a>",2,1,2,0,2022-11-04 20:40:46+00:00
24838,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/24838"">[MRG] Raise NotFittedError when using DictVectorizer without prior fitting</a>",28,0,11,0,2023-04-24 06:28:16+00:00
24903,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/24903"">FIX pairwise_distances not working parallel with custom metric (#24896)</a>",47,8,9,0,2022-12-21 17:06:58+00:00
25010,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25010"">MAINT Parameters validation for decomposition.dict_learning_online</a>",38,4,4,0,2022-12-28 16:44:34+00:00
25097,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25097"">FEA \`PairwiseDistancesReductions\`: support for Boolean \`DistanceMetrics\` via stable simultaneous sort</a>",141,121,10,0,2023-02-13 15:42:48+00:00
25101,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25101"">MAINT Refactor \`Splitter\` into a \`BaseSplitter\`</a>",265,173,50,0,2023-01-26 12:20:37+00:00
25115,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25115"">DOC Revisit SVM C scaling example</a>",101,64,14,1,2023-08-12 10:46:43+00:00
25118,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25118"">MAINT Introduce \`BaseTree\` as a base abstraction for \`Tree\`</a>",596,305,14,0,2023-01-23 08:08:23+00:00
25143,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25143"">PCA - Add Loadings Property</a>",4,0,2,0,2022-12-17 04:38:00+00:00
25148,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25148"">[MRG] Implement predict_proba() for OutputCodeClassifier</a>",48,7,29,0,2022-12-23 17:24:24+00:00
25192,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25192"">FIX improve error message when no samples are available in mutual information</a>",28,1,13,0,2023-01-14 19:05:15+00:00
25217,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25217"">FIX fix and improve OutputCodeClassifier</a>",456,108,1,0,2023-01-16 15:17:46+00:00
25228,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25228"">DOC add an example regarding the multiclass strategies</a>",206,0,34,2,2023-05-04 08:55:35+00:00
25275,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25275"">ENH allow shrunk_covariance to handle multiple matrices at once</a>",35,9,31,1,2023-06-07 16:09:39+00:00
25326,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25326"">feat: Support class weight for MLPClassifier</a>",342,36,2,0,2023-04-16 16:11:07+00:00
25330,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25330"">ENH NearestNeighbors-like classes with metric=""nan_euclidean"" does not actually support NaN values</a>",154,11,16,0,2023-02-24 12:04:50+00:00
25350,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25350"">DOC new example for time-series forecasting with lagged features and prediction intervals</a>",439,0,49,1,2023-03-28 17:09:38+00:00
25356,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25356"">[WIP] FEA Online Dictionary Learning with missing values</a>",323,57,0,0,2023-01-12 17:43:10+00:00
25360,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25360"">DOC Example MLPRegressor as an autoencoder</a>",82,0,1,0,2023-01-31 10:25:59+00:00
25379,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25379"">DOC Update IncrementalPCA example to actually use batches</a>",17,5,8,0,2023-03-01 00:02:16+00:00
25383,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25383"">DOC document about the divergent factor of the Binomial loss</a>",1,0,8,0,2023-08-03 17:47:31+00:00
25434,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25434"">Add \`_asarray_fn\` override to \`check_array\`</a>",57,6,6,0,2023-01-19 16:07:42+00:00
25440,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25440"">[MRG] Fix to text_analytics/fetch_data.py not generating files</a>",33,36,1,0,2023-02-28 23:20:49+00:00
25448,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25448"">MAINT Refactor Tree Cython class to support modularity</a>",1027,443,2,0,2023-01-26 19:18:40+00:00
25462,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25462"">FEA add newton-lsmr solver to LogisticRegression and GLMs</a>",1763,84,43,0,2023-06-27 17:13:37+00:00
25535,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25535"">[DRAFT] Engine plugin API and engine entry point for Lloyd's KMeans</a>",47937,23399,18,0,2023-08-04 08:44:07+00:00
25579,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25579"">MAINT Parameters validation for  \`sklearn.manifold.spectral_embedding\`</a>",13,6,4,0,2023-06-28 14:51:54+00:00
25581,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25581"">TST Add tests to cover errors raised when invalid args passed (sklearn/manifold/_locally_linear)</a>",130,97,2,0,2023-02-11 17:38:26+00:00
25602,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25602"">DOC improve EDA for \`plot_cyclical_feature_engineering.py\`</a>",36,1,9,1,2023-03-28 17:10:19+00:00
25605,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25605"">MNT [engine-api] revert NotSupportedByEngineError feature and automatic catch in unit tests</a>",3,27,0,0,2023-02-14 09:41:10+00:00
25606,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25606"">MAINT Group all sorting utilities in \`sklearn.utils._sorting\`</a>",282,194,23,3,2023-03-24 16:45:40+00:00
25617,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25617"">FEA Add a private \`check_array\` with additional parameters</a>",178,11,11,0,2023-05-04 09:08:34+00:00
25639,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25639"">FEA Implementation of ""threshold-dependent metric per threshold value"" curve</a>",368,0,10,0,2023-05-17 02:11:26+00:00
25645,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25645"">DOC include note for searching for optimal parameters with successive halving</a>",5,0,0,0,2023-02-19 18:18:23+00:00
25646,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25646"">feat: Support sample weight for MLP</a>",345,40,17,0,2023-05-22 04:48:10+00:00
25682,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25682"">DOC Improve narrative of plot_cluster_comparison.py example</a>",92,56,18,0,2023-03-14 15:36:53+00:00
25689,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25689"">FIX Multioutput estimators fail to return raw values</a>",17,14,0,0,2023-02-27 15:46:43+00:00
25714,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25714"">DOC Add demo on parallelization with context manager using different backends</a>",226,0,24,0,2023-03-22 18:28:03+00:00
25789,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25789"">API Gaussian Process: change default behavior of \`sample_y\`</a>",43,5,17,1,2023-07-13 10:43:22+00:00
25807,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25807""> [MRG] FEA ICE lines individually colored by feature values</a>",277,7,0,0,2023-04-18 08:39:19+00:00
25845,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25845"">MinMaxScaler output datatype</a>",55,5,7,0,2023-03-20 09:03:42+00:00
25878,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25878"">DOC Rework outlier detection estimators example</a>",396,149,34,0,2023-06-23 09:10:23+00:00
25886,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25886"">TST use global_random_seed in sklearn/covariance/tests/test_graphical_lasso.py</a>",7,7,1,1,2023-05-02 10:43:03+00:00
25891,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25891"">TST random seed global /svm/tests/test_svm.py</a>",199,126,3,0,2023-04-16 17:59:21+00:00
25894,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25894"">TST test_impute use global random seed</a>",100,76,6,0,2023-04-04 07:12:08+00:00
25939,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25939"">ENH add from_cv_results in RocCurveDisplay</a>",859,82,15,0,2023-06-16 06:27:18+00:00
25948,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25948"">FIX minimum norm solution of unpenalized ridge / OLS when fit_intercept=True</a>",216,82,47,0,2023-04-29 14:27:48+00:00
25977,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25977"">[MRG] DOC GPs: log_marginal_likelihood() log(theta) input</a>",61,17,1,0,2023-06-16 16:18:16+00:00
25978,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25978"">Typo in the range of n_components in PLSRegression</a>",3,3,5,0,2023-03-28 08:15:35+00:00
25991,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/25991"">ENH FeatureUnion: Add verbose_feature_names_out parameter</a>",151,11,16,1,2023-04-29 12:39:33+00:00
26008,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26008"">#23112: showcase idea to cache last step</a>",44,12,0,0,2023-03-29 12:57:30+00:00
26012,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26012"">ENH add verbose option for gpr</a>",133,12,1,0,2023-07-13 10:39:51+00:00
26032,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26032"">[ENH, DRAFT] Implement Precomputed distances in the Pairwise Distances Reduction framework</a>",68,1,6,0,2023-07-27 13:53:27+00:00
26038,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26038"">Update documentation with new non-overlapping patch extraction function</a>",86,0,0,0,2023-03-31 15:27:07+00:00
26094,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26094"">FIX GroupShuffleSplit raises a ValueError for NaN</a>",37,3,8,0,2023-04-13 13:41:55+00:00
26115,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26115"">ENH Support dataframe exchange protocol in ColumnTransformer as input</a>",112,35,8,0,2023-05-31 09:41:15+00:00
26120,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26120"">FEA add TunedThresholdClassifier meta-estimator to post-tune the cut-off threshold</a>",3279,51,111,0,2023-08-07 15:06:51+00:00
26163,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26163"">PERF Improve runtime for early stopping in HistGradientBoosting</a>",145,51,13,1,2023-07-24 19:11:25+00:00
26176,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26176"">fix(areaScores): consistency b/w aupr auroc</a>",5,2,1,0,2023-07-28 05:06:30+00:00
26180,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26180"">DOC TST Add example recommender system</a>",2,0,4,0,2023-08-09 12:48:15+00:00
26189,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26189"">ENH reuse parent histogram in HGBT</a>",75,4,7,0,2023-04-17 17:33:22+00:00
26192,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26192"">Add sampling uncertainty on precision-recall and ROC curves</a>",346,3,5,0,2023-05-11 14:26:33+00:00
26202,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26202"">[mrg] ENH Add custom_range argument for partial dependence - version 2</a>",471,66,4,0,2023-06-21 19:19:40+00:00
26204,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26204"">Fix PLSR n_components documentation.</a>",2,3,7,0,2023-04-19 14:11:42+00:00
26208,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26208"">Pass optional column names to parallel tree function in using the self.feature_names_in attribute.</a>",12,2,7,0,2023-04-18 14:51:37+00:00
26221,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26221"">DOC Rework permutation importance with multicollinearity example</a>",107,57,10,1,2023-08-12 11:00:08+00:00
26243,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26243"">ENH Add Array API compatibility to MinMaxScaler</a>",172,16,48,3,2023-08-07 18:03:56+00:00
26253,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26253"">FIX Raises an erorr in vectorizers when output is pandas</a>",33,0,3,0,2023-04-26 14:36:45+00:00
26266,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26266"">TST Interaction between \`class_weight\` and \`sample_weight\`</a>",215,0,22,0,2023-05-18 18:28:51+00:00
26268,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26268"">ENH Support categories with cardinality higher than max_bins in HistGradientBoosting</a>",300,79,17,1,2023-04-26 14:20:13+00:00
26278,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26278"">ENH replace loss module Gradient boosting</a>",748,1466,40,0,2023-08-12 09:01:58+00:00
26284,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26284"">ENH Add per feature max_categories for OrdinalEncoder</a>",380,20,30,0,2023-08-12 22:11:38+00:00
26299,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26299"">ENH \`learning_curve\` raises a warning on failure of CV folds</a>",24,0,6,1,2023-06-02 21:18:19+00:00
26311,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26311"">FEA HGBT add  post fit calibration for non-canonical link functions</a>",82,6,0,0,2023-04-30 16:24:54+00:00
26330,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26330"">ENH Add \`sample_weight\` parameter to \`OneHotEncoder\`'s \`.fit\`</a>",222,23,1,0,2023-06-23 04:15:37+00:00
26335,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26335"">ENH \`check_classification_targets\` raises a warning when unique classes > 50% of \`n_samples\`</a>",67,10,9,0,2023-06-18 08:47:05+00:00
26353,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26353"">ENH Added stride parameter to extract_patches_2d and reconstruct_from_patches_2d functions.</a>",67,17,0,0,2023-05-10 13:33:49+00:00
26365,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26365"">DOC improve example by reusing best_estimator_</a>",7,4,0,0,2023-05-14 14:51:14+00:00
26366,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26366"">ENH fix x- and y-axis limits and ratio in ROC and PR displays</a>",66,48,2,1,2023-06-16 07:05:43+00:00
26367,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26367"">ENH \`despine\` keyword for ROC and PR curves</a>",114,1,4,0,2023-07-24 08:29:12+00:00
26403,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26403"">[MRG] TST use global_random_seed in sklearn/decomposition/tests/test_pca.py</a>",94,66,10,0,2023-08-02 05:33:38+00:00
26410,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26410"">ENH \`KNeighborsClassifier\` raise when all neighbors of some sample have zero weights</a>",45,1,6,0,2023-06-14 11:55:04+00:00
26411,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26411"">ENH Adds native pandas categorical support to gradient boosting</a>",328,107,35,2,2023-06-23 11:29:51+00:00
26423,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26423"">Feature Stratified k-fold iterators for splitting multilabel data</a>",787,101,0,0,2023-06-14 11:36:56+00:00
26437,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26437"">ENH Speed up make_spd_matrix and allow to generate several matrices</a>",37,18,0,0,2023-06-06 20:11:43+00:00
26440,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26440"">ENH Adds lazy loading to all modules</a>",747,660,18,1,2023-07-26 20:32:37+00:00
26445,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26445"">docs: Improve Perceptron classifier documentation</a>",58,0,6,0,2023-06-02 21:02:33+00:00
26459,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26459"">fix bugs in _correct_predecessor in optics</a>",1,1,9,0,2023-08-14 18:46:40+00:00
26462,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26462"">[MRG] TST use global_random_seed in sklearn/utils/tests/test_optimize.py</a>",2,2,1,0,2023-08-02 05:31:12+00:00
26480,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26480"">FIX Backwards \`SequentialFeatureSelector\` always drops one feature</a>",78,13,3,0,2023-06-06 13:16:32+00:00
26489,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26489"">ENH Add linear, polynomial, sigmoid kernel in example of NuSVC model</a>",43,24,1,0,2023-06-06 12:07:15+00:00
26517,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26517"">DOC Error in the notation of the regularization penalties for multinomial logistic regression in the docs</a>",1,1,4,0,2023-07-26 15:59:40+00:00
26521,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26521"">FIX make sure the decision function of weak learner is symmetric</a>",61,3,13,1,2023-07-13 12:25:33+00:00
26568,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26568"">MAINT Parameter validation for sklearn.neighbors.neighbors_graph</a>",38,17,32,0,2023-07-17 10:02:46+00:00
26574,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26574"">Add support for bools in SimpleImputer (#26292)</a>",46,26,0,0,2023-06-14 00:58:47+00:00
26592,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26592"">[DOC] Improve tree documentation</a>",38,4,17,1,2023-08-14 17:00:24+00:00
26593,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26593"">Fix for Multiclass SVC.fit fails if sample_weight zeros out a class</a>",35,2,3,0,2023-06-22 16:09:17+00:00
26597,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26597"">DOC Fixing the formula for support fraction in Parameters</a>",3,3,1,0,2023-06-20 08:37:58+00:00
26616,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26616"">ENH improve visual HTML estimator representation</a>",687,199,58,1,2023-07-10 07:41:12+00:00
26619,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26619"">DOC Add dropdowns to Module 2.3 Clustering</a>",701,561,9,0,2023-07-24 14:26:56+00:00
26623,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26623"">DOC Add dropdowns to module 1.1 Linear Models</a>",307,223,17,1,2023-07-12 14:59:51+00:00
26648,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26648"">FIX Param validation Interval error for large integers</a>",35,7,11,2,2023-06-23 13:22:02+00:00
26651,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26651"">[WIP] FIX Add tests for pyarrow dtypes in pandas Dataframes</a>",662,117,2,0,2023-06-27 09:55:33+00:00
26654,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26654"">DOC Added dropdowns to Module #1.6 Nearest Neighbors</a>",93,36,5,0,2023-07-12 09:42:57+00:00
26662,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26662"">[MRG] DOC Add dropdowns to Module 1.13 Feature Selection</a>",38,27,7,1,2023-07-26 08:46:12+00:00
26674,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26674"">ENH Allows multiclass target in \`TargetEncoder\`</a>",341,60,40,1,2023-08-09 23:58:16+00:00
26675,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26675"">ENH Feature/classification report with preds</a>",3218,1967,1,0,2023-07-06 09:07:56+00:00
26683,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26683"">ENH Adds polars support to ColumnTransformer</a>",375,62,25,1,2023-07-30 16:41:26+00:00
26686,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26686"">MAINT Move stuff outside of utils.__init__</a>",1208,1186,6,0,2023-07-06 14:49:05+00:00
26687,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26687"">Small doc change to help people finding (via browser search) content in a fold</a>",4,1,2,1,2023-06-27 13:14:54+00:00
26689,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26689"">MRG Add decision_function, predict_proba and predict_log_proba for NearestCentroid estimator</a>",208,14,2,0,2023-06-29 15:28:12+00:00
26691,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26691"">ENH run common tests for SparseCoder</a>",244,12,1,0,2023-07-10 09:57:39+00:00
26692,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26692"">[MRG] MAINT Allow estimators to opt-in to complex numbers in \`check_array\`</a>",40,5,18,0,2023-07-08 23:15:57+00:00
26693,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26693"">DOC Adding Dropdown to module 7.2 Realworld Datasets</a>",19,10,8,0,2023-08-01 16:53:21+00:00
26694,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26694"">DOC Adding dropdown for module 2.1 Gaussian Mixtures</a>",72,60,22,0,2023-08-03 09:43:44+00:00
26699,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26699"">Doc Add dropdowns to 1.10.decision trees</a>",29,7,2,1,2023-07-27 15:53:43+00:00
26701,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26701"">DOC improve section folding behaviour</a>",42,7,11,1,2023-07-05 19:14:16+00:00
26707,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26707"">BENCH newton-lsmr solver for GLMs - inner stopping criterion</a>",11219,87,1,0,2023-07-02 12:49:06+00:00
26710,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26710"">DOC Add dropdowns to module 7.1 Toy datasets</a>",82,62,4,1,2023-07-27 08:46:20+00:00
26713,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26713"">FIX samples average computation for multi-label classification</a>",126,3,1,0,2023-06-27 13:53:11+00:00
26720,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26720"">DOC Adding dropdown for module 2.2 Manifold Learning</a>",50,20,3,1,2023-07-29 12:20:13+00:00
26721,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26721"">WIP 1/n * loss in LinearModelLoss</a>",115,62,2,0,2023-08-10 10:13:17+00:00
26729,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26729"">FIX do not allow for p=None in NN-based algorithm</a>",2,2,3,0,2023-06-29 13:15:52+00:00
26734,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26734"">API replace mean_squared_error(square=False) by root_mean_squared_error</a>",307,25,80,1,2023-07-17 10:20:38+00:00
26735,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26735"">GaussianMixture with BIC/AIC</a>",507,41,3,0,2023-07-25 21:20:58+00:00
26736,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26736"">[ENH] Adding estimators_samples_ for forest models</a>",177,2,14,0,2023-08-14 17:02:55+00:00
26746,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26746"">Simplified computation for pair confusion matrix</a>",4,4,2,0,2023-07-26 14:54:16+00:00
26757,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26757"">DOC Add dropdowns to module 2.8. Density Estimation</a>",6,0,3,1,2023-07-24 14:30:51+00:00
26805,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26805"">DOC Notebook style and enhanced descriptions in SVM kernels example</a>",283,73,48,2,2023-08-01 14:29:20+00:00
26807,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26807"">DOC Added dropdowns to 6.2 feature-extraction</a>",35,9,10,1,2023-07-28 08:40:39+00:00
26809,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26809"">DOC Update web site to pydata-sphinx theme.</a>",945,543,35,0,2023-07-14 18:29:29+00:00
26814,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26814"">MNT Remove DeprecationWarning for scipy.sparse.linalg.cg tol vs rtol argument</a>",17,4,8,1,2023-08-09 09:39:11+00:00
26819,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26819"">DOC: added dropdowns to module 1.9 naive bayes</a>",15,2,6,2,2023-07-25 10:02:44+00:00
26820,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26820"">PERF speed up confusion matrix calculation</a>",85,18,7,0,2023-07-25 15:25:59+00:00
26828,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26828"">PERF Implement PairwiseDistancesReduction backend for RadiusNeighbors predict_proba</a>",573,2,79,0,2023-08-13 18:27:01+00:00
26830,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26830"">MNT Deprecate SAMME.R algorithm from AdaBoostClassifier</a>",75,197,23,1,2023-08-01 14:53:17+00:00
26837,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26837"">Bug fix.</a>",17,12,2,0,2023-07-17 07:13:56+00:00
26839,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26839"">ENH add pos_label to confusion_matrix</a>",130,18,26,1,2023-08-12 10:53:41+00:00
26840,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26840"">ENH add new response_method in make_scorer</a>",164,36,18,0,2023-08-03 13:35:44+00:00
26860,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26860"">MAINT create y before enforcing X using tags</a>",12,11,7,0,2023-07-24 15:05:16+00:00
26877,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26877"">DOC Add links to preprocessing examples in docstrings and userguide</a>",76,50,19,1,2023-08-04 15:15:17+00:00
26881,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26881"">DOC Add dropdowns to module 9.1 Python specific serialization</a>",183,169,2,1,2023-07-25 09:15:25+00:00
26882,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26882"">DOC Add dropdowns to module 9.2 Interoperable formats</a>",10,0,2,0,2023-07-25 09:21:06+00:00
26884,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26884"">MAINT Parameters validation for utils.extmath.weighted_mode</a>",9,0,1,0,2023-07-28 19:16:04+00:00
26886,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26886"">CI Add Python 3.12 build</a>",81,8,4,0,2023-08-07 12:11:37+00:00
26888,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26888"">ENH Add \`np.float32\` data support for \`HDBSCAN\`</a>",44,16,1,0,2023-07-24 17:41:43+00:00
26904,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26904"">DOC Add links to \`pipelines\` examples in docstrings</a>",17,2,13,0,2023-08-01 13:48:44+00:00
26913,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26913"">FIX CalibratedClassifierCV with sigmoid and large confidence scores</a>",58,1,35,0,2023-08-11 07:48:05+00:00
26919,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26919"">DOC Sparse martix documentation in OneHotEncoder and errormessage for mismatching output formats in transformers</a>",19,8,9,0,2023-08-07 16:42:01+00:00
26923,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26923"">MRG Weighted v_measure_score (and related functions: entropy, contingency matrix & mutual info score)</a>",184,20,4,0,2023-08-09 12:58:32+00:00
26924,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26924"">DOC Add dropdowns to preprocessing.rst</a>",3,1,2,0,2023-07-29 03:18:38+00:00
26926,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26926"">DOC add link to sklearn_is_fitted example in check_is_fitted</a>",5,2,1,0,2023-07-28 13:41:47+00:00
26932,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26932"">DOC Add links to decomposition examples in docstrings and user guide</a>",17,0,11,0,2023-08-02 16:51:47+00:00
26934,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26934"">DOC add links to cross decomposition examples</a>",9,0,9,0,2023-08-02 12:07:45+00:00
26935,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26935"">DOC add links to neural network examples</a>",14,1,12,0,2023-08-02 17:30:56+00:00
26944,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26944"">ENH Improve warnings if func returns a dataframe in FunctionTransformer</a>",45,15,1,0,2023-08-12 10:54:51+00:00
26946,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26946"">DOC adds dropdown for 10.3 Controlling Randomness</a>",4,0,1,0,2023-08-01 01:41:50+00:00
26950,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26950"">DOC Add example links for feature_selection.RFE</a>",2,0,2,0,2023-07-31 11:50:44+00:00
26951,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26951"">DOC Add links to plot_document_clustering example</a>",8,0,4,1,2023-08-03 11:09:27+00:00
26952,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26952"">DOC clearer definition of estimator to be used in last step of a pipeline</a>",26,19,12,0,2023-08-07 15:44:41+00:00
26953,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26953"">DOC Add returned unit to mutual information documentation</a>",10,8,1,0,2023-08-04 12:50:40+00:00
26956,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26956"">DOC Notebook style for ClassifierChain example</a>",67,47,3,0,2023-08-01 09:53:01+00:00
26958,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26958"">TST Fix typo, lint \`test_target_encoder.py\`</a>",19,11,15,2,2023-08-08 04:06:17+00:00
26962,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26962"">DOC Add link to plot_tree_regression.py example</a>",3,0,2,0,2023-08-02 14:25:20+00:00
26967,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26967"">DOC Add links to text/plot_hashing_vs_dict_vectorizer.py example</a>",15,0,3,0,2023-08-12 14:15:35+00:00
26969,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26969"">DOC added link to example plot_svm_margin.py</a>",4,0,2,0,2023-08-02 08:39:32+00:00
26970,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26970"">DOC Add link to plot_optics.py</a>",4,0,10,0,2023-08-09 21:49:51+00:00
26971,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26971"">DOC linked SplineTransformer to time-related feature engineering example</a>",3,0,3,1,2023-08-11 22:37:45+00:00
26972,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26972"">Added example to sklearn.svm.NuSVC</a>",3,0,3,0,2023-08-11 22:42:33+00:00
26973,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26973"">DOC improve iris example</a>",51,31,21,0,2023-08-12 14:55:25+00:00
26974,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26974"">add link to example cluster_iris</a>",3,0,2,0,2023-08-02 10:17:06+00:00
26975,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26975"">DOC add link to plot_isolation_forest.py</a>",3,0,4,1,2023-08-11 21:18:52+00:00
26976,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26976"">FIX Create a cosistent image via random_state additions in ""plot_cluster_comparison.py""</a>",19,8,9,0,2023-08-11 22:27:41+00:00
26978,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26978"">DOC Add link to plot_classification_probability.py</a>",63,14,28,0,2023-08-02 13:29:29+00:00
26983,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26983"">FEA Introduce \`PairwiseDistances\`, a generic back-end for \`pairwise_distances\`</a>",629,54,3,0,2023-08-04 09:44:00+00:00
26991,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26991"">DOC Add example showcasing HGBT regression</a>",483,9,1,0,2023-08-03 09:29:10+00:00
26995,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/26995"">ENH handle mutliclass with scores and probailities in DecisionBoundaryDisplay</a>",118,42,5,0,2023-08-03 12:58:09+00:00
27002,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/27002"">FIX accept multilabel-indicator in _get_response_values</a>",94,18,9,0,2023-08-12 10:44:42+00:00
27003,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/27003"">Add concept of ""soft deprecation"" to development tools</a>",29,0,8,0,2023-08-14 07:39:04+00:00
27005,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/27005"">ENH add metadata routing to ColumnTransformer</a>",455,99,22,0,2023-08-13 20:07:35+00:00
27018,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/27018"">MAINT Make \`ArgKminClassMode\` accept sparse datasets</a>",14,29,6,2,2023-08-14 06:20:05+00:00
27020,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/27020"">DOC clustering speed with connectivity matrices</a>",1,1,3,1,2023-08-07 18:17:59+00:00
27022,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/27022""> [MRG] DOC Add dropdown to Module 6.1 'Pipelines and composite estimators'  #26617 </a>",63,50,3,0,2023-08-08 08:43:46+00:00
27024,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/27024"">MNT Update lock files</a>",233,240,2,1,2023-08-07 12:53:13+00:00
27025,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/27025"">DOC Add link to Early Stopping example in Gradient Boosting</a>",4,3,1,0,2023-08-07 11:09:45+00:00
27027,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/27027"">CI Build and test Python 3.12 wheels</a>",48,4,9,0,2023-08-09 15:29:20+00:00
27034,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/27034"">DOC take \`Examples\` out of a dropdown (#26617, #26641)</a>",8,6,3,0,2023-08-09 11:30:13+00:00
27040,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/27040"">Split GLM fit in many functions; add LBFGS solver as a class</a>",122,106,2,0,2023-08-10 09:17:18+00:00
27041,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/27041"">WIP: Adapt sklearn for NumPy default integer change</a>",48,24,1,0,2023-08-09 12:39:47+00:00
27042,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/27042"">MNT Adjust code after NEP 51 numpy scalar formatting changes</a>",10,8,5,0,2023-08-10 12:58:05+00:00
27051,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/27051"">FIX Check that \`RadiusNeighborsClassifier\` is fit in \`predict{,_proba}\`</a>",19,0,9,1,2023-08-13 18:35:10+00:00
27052,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/27052"">DOC fix behavior of copy button in installation instructions</a>",92,34,2,1,2023-08-14 14:10:13+00:00
27053,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/27053"">DOC Add missing links to examples/impute (scikit-learn#26927)</a>",4,0,1,0,2023-08-11 15:42:29+00:00
27056,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/27056"">Added links for random_datasets example</a>",9,0,1,0,2023-08-12 15:36:01+00:00
27058,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/27058"">FEAT add metadata routing to *SearchCV</a>",247,56,1,0,2023-08-13 20:05:58+00:00
27064,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/27064"">Doc: Updating getting started and Installation guide changes</a>",8,8,1,0,2023-08-14 01:45:16+00:00
27066,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/27066"">Documents fixed to increase speed by removing for loop </a>",1,1,1,0,2023-08-14 11:46:21+00:00
27067,"<a href=""https://github.com/scikit-learn/scikit-learn/pull/27067"">DOC Remove outdated instructions for Apple Silicon</a>",0,22,1,0,2023-08-14 14:54:25+00:00

"""
)

results_df = pd.read_csv(FILE, parse_dates=["updated"])

multi_choice = pn.widgets.MultiChoice(
    name="Sort values by",
    value=["additions"],
    options=results_df.columns.to_list(),
    margin=(8, 8, 0, 8),
)
ascending = pn.widgets.RadioBoxGroup(
    options=[True, False],
    inline=True,
    margin=(12, 8, 0, 8),
    width=110,
)

selector = pn.Column(
    multi_choice,
    pn.Row(
        pn.pane.Markdown("#### Ascending?", width=70, margin=(8, 8, 8, 8)),
        ascending,
        pn.pane.Markdown(
            f"#### Generated on {DATE} [Source on GitHub](https://github.com/{TRACKER_REPO})",
            margin=(8, 8, 8, 8),
        ),
    ),
    styles=dict(background="WhiteSmoke"),
)


results_df = results_df.interactive()
results2_df = results_df.sort_values(multi_choice, ascending=ascending)


out_df = (
    results2_df.style.hide()
    .format({"updated": "{:%Y-%m-%d}"})
    .set_properties(**{"font-size": "1.25em"})
    .set_table_styles(
        [
            {
                "selector": "th.col_heading",
                "props": "font-size: 1.25em;",
            },
            {"selector": "tr:hover", "props": "cursor: default;"},
        ],
    )
)

pn.Column(
    selector,
    pn.pane.DataFrame(out_df, escape=False),
).servable()


await write_doc()
  `

  try {
    const [docs_json, render_items, root_ids] = await self.pyodide.runPythonAsync(code)
    self.postMessage({
      type: 'render',
      docs_json: docs_json,
      render_items: render_items,
      root_ids: root_ids
    })
  } catch(e) {
    const traceback = `${e}`
    const tblines = traceback.split('\n')
    self.postMessage({
      type: 'status',
      msg: tblines[tblines.length-2]
    });
    throw e
  }
}

self.onmessage = async (event) => {
  const msg = event.data
  if (msg.type === 'rendered') {
    self.pyodide.runPythonAsync(`
    from panel.io.state import state
    from panel.io.pyodide import _link_docs_worker

    _link_docs_worker(state.curdoc, sendPatch, setter='js')
    `)
  } else if (msg.type === 'patch') {
    self.pyodide.globals.set('patch', msg.patch)
    self.pyodide.runPythonAsync(`
    state.curdoc.apply_json_patch(patch.to_py(), setter='js')
    `)
    self.postMessage({type: 'idle'})
  } else if (msg.type === 'location') {
    self.pyodide.globals.set('location', msg.location)
    self.pyodide.runPythonAsync(`
    import json
    from panel.io.state import state
    from panel.util import edit_readonly
    if state.location:
        loc_data = json.loads(location)
        with edit_readonly(state.location):
            state.location.param.update({
                k: v for k, v in loc_data.items() if k in state.location.param
            })
    `)
  }
}

startApplication()