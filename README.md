# TAB-GAN: A Transformer-Augmented Bidirectional GRU Generative Adversarial Network for Stock Price Prediction

This project is the implementation of the master's thesis "A Transformer-Augmented Bidirectional GRU Generative Adversarial Network for Stock Price Prediction". The thesis proposes the TAB-GAN model to address the limitations of existing models in capturing long-term dependencies in time series data and in effectively processing conditional information.

## Abstract

With the advancement of deep learning technologies, neural networks have demonstrated significant potential in financial market analysis. However, existing models still face challenges in adequately capturing long-term dependencies in time series data and in effectively processing conditional information. This study aims to address these limitations by proposing a Transformer-Augmented Bidirectional GRU Generative Adversarial Network (TAB-GAN) model. Through a multi-level feature extraction strategy and the Transformer architecture, the model enhances the ability to capture long-term dependencies and improve conditional information processing.

In the empirical study, this research uses Prophet and DCC-GARCH models to establish test datasets and analyzes five publicly listed companies from various industries. The experimental results indicate that TAB-GAN achieves 30–50% lower prediction errors compared to other models and demonstrates superior distribution learning capabilities in terms of the KL divergence metric.

In the application of trading strategies, this study analyzes five Taiwan-listed stocks by establishing prediction intervals based on different confidence levels (50%, 70%, 90%) and comparing them with traditional Bollinger Bands strategy. The results indicate that the 50% confidence level strategy performs optimally in upward trending markets, achieving an annualized return of 383.97%, while the 90% confidence level strategy demonstrates stable performance in consolidating and downward trending markets. In terms of risk-adjusted returns, TAB-GAN's Sharpe ratios consistently outperform traditional Bollinger Bands strategy, demonstrating TAB-GAN's ability to effectively balance returns and risks.

## Citation

If you find this project useful, please cite the following thesis:

```
Tsai, Zhi-Hong. Transformer Augmented Bidirectional GRU Generative Adversarial Network and Corresponding Financial Investment Strategies. 2025. Fu Jen Catholic University, Master’s Thesis. National Digital Library of Theses and Dissertations in Taiwan. https://hdl.handle.net/11296/2j27h8
```

## Features

*   **Multiple GAN Models:** Implements and compares three GAN architectures:
    *   **ForGAN:** A GAN model based on LSTM or GRU cells.
    *   **RCGAN:** A recurrent conditional GAN.
    *   **TAB-GAN:** The main model of this project, a Transformer-Augmented Bidirectional GRU GAN.
*   **Data Processing:** A comprehensive data processing pipeline that includes:
    *   Loading and cleaning of historical stock data.
    *   Feature engineering with various technical indicators.
    *   Data scaling and splitting into training and testing sets.
*   **Hyperparameter Optimization:** Utilizes Bayesian optimization (with Optuna) to find the optimal hyperparameters for the GAN models.
*   **Model Evaluation:** Provides scripts to evaluate the models' predictive power and the quality of the generated data.
*   **Trading Strategy:** Includes a module to generate trading signals and backtest trading strategies based on the model's predictions.
*   **R Integration:** Uses the `rmgarch` package in R for simulating financial data, with `rpy2` as the interface.

## Getting Started

### Prerequisites

*   Python 3.9+
*   Conda
*   R environment with the `rmgarch` package installed.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/TAB-GAN.git
    cd TAB-GAN
    ```

2.  **Create and activate the Conda environment:**
    ```bash
    conda create --name tabgan --file requirements.txt
    conda activate tabgan
    ```

3.  **Install R and `rmgarch`:**
    Make sure you have a working R installation. Then, open an R console and run:
    ```R
    install.packages("rmgarch")
    ```

## Usage

The main entry point of the project is `main.py`. You can run different modes by specifying the `--mode` argument.

*   **Train a model:**
    ```bash
    python main.py --mode train --model tabgan --stock 2330 --cuda 0
    ```

*   **Run Bayesian optimization:**
    ```bash
    python main.py --mode optim --model tabgan --stock 2330 --cuda 0
    ```

*   **Test a trained model:**
    ```bash
    python main.py --mode test --model tabgan --stock 2330 --cuda 0
    ```

*   **Generate simulated data:**
    ```bash
    python main.py --mode simulate --stock 2330
    ```

### Arguments

*   `--mode`: Processing mode (`train`, `optim`, `test`, `simulate`).
*   `--model`: Model to use (`forgan`, `rcgan`, `tabgan`).
*   `--stock`: Stock symbol (e.g., `2330`).
*   `--cuda`: CUDA device number to use.
*   Other arguments are defined in `arguments.py`.

## File Structure

```
/
├───data/                 # Stock data and simulated data
├───lib/                  # Helper modules for data processing, visualization, etc.
│   ├───data.py           # Data processing and Dataset classes
│   ├───calc.py           # Calculation of technical indicators and metrics
│   └───...
├───model/                # Model implementations
│   ├───tabgan.py         # TAB-GAN model
│   ├───algos/            # Model architectures
│   └───...
├───arguments.py          # Argument parsing
├───bayes_optim.py        # Bayesian optimization script
├───eval.py               # Model evaluation script
├───main.py               # Main execution script
├───requirements.txt      # Python dependencies
├───strategy.py           # Trading strategy and signal generation
└───train.py              # Model training script
```

## Dependencies

The main dependencies are listed in `requirements.txt`. Key libraries include:

*   `pytorch`
*   `pandas`
*   `numpy`
*   `scikit-learn`
*   `optuna`
*   `rpy2`
*   `matplotlib`
*   `seaborn`
*   `prophet`