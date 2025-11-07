# NeurIPS 论文趋势分析与预测 (NeurIPS Paper Trend Analysis and Prediction)

一个关于 NeurIPS 论文集(1987-2017)的趋势分析项目。使用 NMF 进行主题建模，并使用随机森林预测未来的研究热点。

(An analysis of NeurIPS paper trends (1987-2017) using NMF topic modeling and Random Forest to predict future hot topics.)

---

## Project Workflow (项目工作流)

This project contains three Jupyter Notebooks that form a complete pipeline for processing, clustering, and trend prediction of NeurIPS paper data.

The project notebooks **must** be executed in the following order:

1.  **`processing.ipynb`**
    * **Input:** Raw CSV files from the `../dataset/` directory.
    * **Output:** Generates several `.pkl` (pickle) files, such as `tfidf_matrix.pkl`, `nmf_model.pkl`, and `processed_papers_WITH_TOPICS.pkl`.

2.  **`clustering.ipynb`**
    * **Input:** Loads the `.pkl` files created by `processing.ipynb`.
    * **Output:** Performs clustering and analysis, then saves the final labeled dataset as a `.csv` file (e.g., `df_final_with_topics.csv`).

3.  **`Analysis.ipynb`** (或 `trend_prediction.ipynb`，请根据您的实际文件名修改)
    * **Input:** Loads the `.csv` file created by `clustering.ipynb`.
    * **Output:** Conducts and visualizes time-series trend analysis on the clustered topics.

---

## How to Run This Project (如何运行)

> ！！！**Problem:** Pickle files (.pkl) are not portable across different Python environments, especially if the versions of libraries like numpy, pandas, or scikit-learn do not match exactly.

> ！！！Attempting to run `clustering.ipynb` in a new environment using `.pkl` files from an old one will likely fail (often with errors like `No module named 'numpy._core.numeric'`).

### Required Execution Steps (必需的执行步骤)
1.  **Set Up Datasets**: Place your raw CSV data files (e.g., `papers.csv`) inside a `../dataset/` directory relative to the notebooks.

2.  **Install Dependencies**: Run the following command in terminal to make sure you have installed all dependencies.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run Processing**: You must first run the entire `processing.ipynb` notebook in your new environment. This will generate the environment-compatible `.pkl` files required by the next step. **Note: This step can be very time-consuming.**

4.  **Run Clustering**: Once the `.pkl` files have been generated locally, run `clustering.ipynb`. This will load those files, perform clustering, and save the final `.csv` file.

5.  **Run Prediction**: You can now run `Analysis.ipynb` (或 `trend_prediction.ipynb`), which will load the stable `.csv` file and perform the trend analysis.

---

## Environment and Dependencies (环境与依赖)

This project was originally developed in Google Colab. To ensure portability and successful execution in a new environment, maintaining consistent package versions is critical.

### Key Dependencies

This project relies on the following Python libraries:

* `pandas`
* `numpy`
* `scikit-learn`
* `nltk`
* `tqdm`
* `gensim`
* `matplotlib`
* `seaborn`
* `statsmodels`
* `prophet`

### `requirements.txt` (Crucial)

To guarantee version consistency, it is strongly recommended to generate a `requirements.txt` file from your original, working (e.g., Colab) environment.

---

## Project Report: Analysis Summary (项目报告：分析总结)

### 1. Introduction/Background (引言/背景)

* **研究问题:**
    本项目的主要目标是分析 NeurIPS 论文数据集中隐藏的主题趋势，并构建一个预测模型，以预测下一年度最热门的5个研究主题。
* **研究动机:**
    在人工智能领域，研究热点变化迅速。能够通过历史数据挖掘主题演变规律，并预测未来的研究趋势，对于科研人员、学生和研究机构制定研究方向具有重要的指导意义和实用价值。

### 2. Methods (方法)

* **数据探索:**
    数据集包含1987年至2017年的 NeurIPS 论文。我们首先对数据进行了合并（论文与作者信息）和清洗 (`processing.ipynb`)。
* **分析流程:**
    * **数据预处理 (Preprocessing):** 对论文文本进行标准化处理，包括去除停用词、标点符号，并使用词形还原（Lemmatization）来统一词义 (`processing.ipynb`)。
    * **特征提取 (Feature Extraction):** 使用 TF-IDF 来提取关键词特征（包括 unigrams 和 bigrams） (`processing.ipynb`)。
    * **主题建模 (Topic Modeling):**
        * **NMF (非负矩阵分解):** 我们主要使用 NMF 将 TF-IDF 矩阵分解为**20个**潜在主题。主题数量（K=20）是通过遍历K值并评估匹配度来确定的 (`processing.ipynb`)。
        * **LDA (潜在狄利克雷分配):** 作为对比，我们也尝试了 LDA 模型（9个主题）。
    * **主题验证 (Topic Validation):**
        * 为了验证 NMF 聚类结果的合理性，我们使用**逻辑回归**模型，以 TF-IDF 特征为X，NMF分配的主题ID为Y进行训练。该模型达到了约 **87%** 的准确率 (`clustering.ipynb`)。
        * LDA 的9个主题验证准确率也达到 86%，但 NMF 的20个主题分类效果更优，因此后续分析基于 NMF。
    * **趋势预测模型 (Trend Prediction Model):**
        * 我们使用**随机森林分类器 (Random Forest Classifier)** 来预测一个主题在下一年是否会成为“热门主题”（定义为年度 Top 5 主题） (`Analysis.ipynb`)。
        * **特征工程:** 我们构建了基于时间序列的特征，包括：该主题近年的论文份额 (`share_t`)、3年滑动平均 (`share_t_ma3`)、份额增长加速度 (`acceleration`)、5年趋势斜率 (`trend_slope_5y`) 以及“核心作者”在该主题的活跃度 (`core_author_presence_t`) 和关注度增长 (`core_author_focus_boost_t`) (`Analysis.ipynb`)。

### 3. Result (结果)

* **聚类可视化 (t-SNE):** 我们对 NMF 产生的文档-主题矩阵进行了 t-SNE 降维，可视化结果显示20个主题形成了相对清晰的簇群，表明 NMF 成功地分离了不同的研究领域 (`clustering.ipynb`)。
* **主题趋势 (面积图):** 我们绘制了1987年至2017年各主题论文数量的堆叠面积图。该图直观地展示了研究趋势的变迁，例如“经典神经网络”的衰落，以及“深度学习”和“优化算法”在近十年的迅猛增长 (`clustering.ipynb`)。
* **预测模型性能:**
    * 随机森林分类器在预测“Top 5 热门主题”方面表现出色。在对2013年至2017年进行的滚动回测中，该模型对下一年 Top 5 热门主题的平均**召回率（Recall）达到了 88%** (`Analysis.ipynb`)。
    * 利用截至2017年的所有数据，我们的模型预测2018年的 Top 5 热门主题为：1. 深度学习 (Deep Learning), 2. 优化算法 (Optimization), 3. 贝叶斯推断 (Bayesian Inference), 4. 矩阵/张量方法 (Matrix/Tensor), 5. 图模型 (Graphical Models) (`Analysis.ipynb`)。

### 4. Conclusion/Summary (结论/总结)

* 本实验证明，通过结合 TF-IDF、NMF 主题建模和随机森林分类器，可以构建一个有效的模型来分析和预测 NeurIPS 论文集中的研究热点。
* 分析结果表明，基于“核心作者”活跃度的特征（如 `core_author_presence_t`）是预测未来趋势的强信号。
* 该模型成功捕捉到了“深度学习”等领域的爆炸性增长，并提供了对2018年研究趋势的可靠预测。
