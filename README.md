# FRET-Lib

FRET-Lib is a Rust application for analysing single-molecule Förster resonance energy transfer (smFRET) traces. It provides a graphical workflow for loading traces exported by [iSMS](https://pubmed.ncbi.nlm.nih.gov/26125588/), filtering and truncating them at photobleaching events, fitting a Gaussian-emission Hidden Markov Model (HMM), and inspecting or exporting the resulting idealized traces.

The application is designed to balance automation with manual control. A user can supply an HMM directly, use a learner to estimate its parameters, initialize those parameters automatically, or ask the program to estimate the number of states as well.

> **Current input scope:** the loader expects the text format exported by iSMS. The parsing layer is isolated in `src/trace_selection/trace_loader.rs`, so support for other formats can be added without changing the analysis pipeline.

## Contents

- [How the application works](#how-the-application-works)
- [Getting started](#getting-started)
- [Input data](#input-data)
- [Repository organization](#repository-organization)
- [Signal-analysis pipeline](#signal-analysis-pipeline)
- [Using the crate without the GUI](#using-the-crate-without-the-gui)
- [Example use case](#example-use-case)

## How the application works

FRET-Lib follows three main stages in order:

1. **Load data** — read one iSMS trace file or every compatible file in a directory.
2. **Preprocess traces** — extract FRET-efficiency sequences, optionally apply trace-quality tests, and detect photobleaching so post-bleach samples can be excluded.
3. **Analyse signals** — configure and run an HMM pipeline, inspect the inferred states and idealized traces, and export a report.

The GUI exposes these stages through three tabs:

- **Main** contains trace loading, preprocessing, HMM pipeline configuration, and a console-style activity log.
- **Raw Analysis** becomes available after traces are loaded. It plots individual FRET traces and a combined histogram, and updates after preprocessing.
- **Analysis** becomes available after a successful HMM run. It shows state parameters, occupancies, probability matrices, Gaussian state distributions, and idealized traces overlaid on the observations.

## Getting started

### Requirements

- A recent stable [Rust toolchain](https://www.rust-lang.org/tools/install)
- A desktop environment capable of running an `eframe`/`egui` native application

### Run from source

```bash
git clone <repository-url>
cd fret-lib
cargo run --release
```

The release profile is recommended for computationally expensive learning and state-count searches. For development, `cargo run` gives faster builds.

### Build and test

```bash
cargo build --release
cargo test
```

The repository also contains `setup.sh`, which builds a release binary, copies it to `~/.local/bin`, and updates Bash and Zsh PATH configuration when required. Review the script before running it because it changes files in your home directory.

## Input data

Each input file represents one FRET pair and must follow the iSMS text-export layout:

1. An unused introductory line.
2. Three metadata lines containing the date, movie filename, and FRET-pair number.
3. A header row naming the available traces.
4. One numeric row per time step, with the same number of values in every column.

Recognized columns include direct, background-corrected, and raw donor/acceptor channels, stoichiometry (`S`), and FRET efficiency (`E`). A FRET-efficiency column is required for signal analysis. Some filtering tests also require a matching donor/acceptor intensity pair; when both raw and background-corrected pairs are present, the corrected pair is preferred.

Multiple files can be loaded together and are analysed as one sequence set. They should therefore describe the same experimental system. An invalid file produces an error, but does not prevent valid files selected in the same operation from being loaded.

Example data is available in [`traces_data_with_intruder/`](traces_data_with_intruder/). As its name suggests, this directory also contains intentionally unsuitable inputs for demonstrating loader validation.

## Repository organization

```text
fret-lib/
├── Cargo.toml                         Package metadata and dependencies
├── Cargo.lock                         Locked dependency versions
├── setup.sh                           Optional local installation helper
├── src/
│   ├── main.rs                        Native application entry point
│   ├── lib.rs                         Public crate modules
│   ├── interface.rs                   GUI module declarations
│   ├── interface/                     egui application, tabs, dialogs, reporting
│   ├── trace_selection.rs             Loading/preprocessing module declarations
│   ├── trace_selection/               Trace model, parsing, filtering, utilities
│   ├── signal_analysis.rs             Signal-analysis module declarations
│   ├── signal_analysis/hmm.rs         HMM public surface and re-exports
│   ├── signal_analysis/hmm/            HMM pipeline and algorithms
│   ├── optimization.rs                Optimization module declarations
│   └── optimization/                  Generic AMaLGaM and numerical machinery
└── traces_data_with_intruder/         Example iSMS-like trace files
```

### Entry points and public surface

| Path | Responsibility |
| --- | --- |
| `src/main.rs` | Launches the native GUI as **FRET Analysis GUI**. It also contains a non-GUI HMM example function for developers. |
| `src/lib.rs` | Exposes the four top-level modules: `interface`, `trace_selection`, `signal_analysis`, and `optimization`. |
| `Cargo.toml` | Declares the linear-algebra, plotting, random-number, GUI, file-dialog, and heatmap dependencies. |

### `src/interface/` — desktop GUI

This layer turns the library components into the end-to-end desktop workflow.

| File | Responsibility |
| --- | --- |
| `app.rs` | Owns shared application state, switches between tabs, and runs the configured HMM. |
| `main_tab.rs` | Coordinates loading, preprocessing, pipeline enablement, settings dialogs, and log messages. |
| `load_traces_window.rs` | Selects individual files or a directory and passes them to the trace loader. |
| `filter_settings_window.rs` | Enables and configures preprocessing tests. |
| `learn_settings_window.rs` | Selects Baum–Welch or AMaLGaM and edits learner-specific settings. |
| `init_settings_window.rs` | Configures state-value, state-noise, start-matrix, and transition-matrix initialization. |
| `nsf_settings_window.rs` | Configures the number-of-states strategy and search bounds. |
| `run_signal_analysis_window.rs` | Collects whichever input the first enabled pipeline component still requires. It also parses and normalizes manually entered probability matrices. |
| `raw_analysis_tab.rs` | Displays loaded/preprocessed traces and the combined value histogram. |
| `analysis_tab.rs` | Displays fitted states, occupancy, matrices, distributions, and idealized sequences. |
| `reporter.rs` | Builds the text report exported from the analysis view. |

Settings dialogs use an explicit **Set** action: closing a dialog without pressing **Set** does not apply the changes.

### `src/trace_selection/` — loading and preprocessing

| File | Responsibility |
| --- | --- |
| `trace_loader.rs` | Parses a single iSMS text export or loads all files in a directory, including metadata and header validation. |
| `individual_trace.rs` | Represents one typed channel and validates its numeric samples. It also stores derived photobleaching/lifetime information. |
| `point_traces.rs` | Groups all channels and metadata belonging to one FRET pair/file. |
| `set_of_points.rs` | Manages all loaded pairs, filtering results, rejected points, and extraction of the FRET sequence set used by the HMM. |
| `filter.rs` | Defines configurable comparisons, the default filter setup, computed filter statistics, and failure reasons. |
| `filters.rs`, `tools.rs` | Supply median filtering, gradients, means/standard deviations, and signal-to-noise calculations used during preprocessing. |

The available quality checks cover photobleaching steps, donor and FRET lifetimes, background and signal SNR, donor/acceptor correlation, background noise, intensity range, first-step FRET, maximum FRET, and average FRET. Tests can be enabled individually. Keeping preprocessing enabled is recommended even with no quality tests selected, because it also performs photobleaching detection.

### `src/signal_analysis/hmm/` — HMM analysis

This directory contains the scientific core of the application.

| Area | Responsibility |
| --- | --- |
| `hmm_struct.rs` | Defines the `HMM` façade, valid component order, input variants, pipeline execution, and top-level errors. |
| `state.rs` | Defines Gaussian states (identifier, mean/value, and standard deviation/noise) and validates them. |
| `hmm_matrices.rs` | Defines and validates start and transition probability matrices. |
| `hmm_instance.rs` | Evaluates a concrete HMM and can generate synthetic state/observation sequences. |
| `probability_matrices.rs` | Computes forward, backward, gamma, and xi probabilities, including scaled variants. |
| `viterbi.rs` | Finds the most likely hidden-state sequence for an observation trace. |
| `baum_welch.rs` | Implements Baum–Welch parameter estimation and its termination modes. |
| `optimization_tracker.rs` | Tracks convergence, iteration limits, plateaus, and collapsed-state handling. |
| `analysis/` | Runs final inference and exposes fitted states, matrices, occupancy, likelihoods, and idealized sequences to the GUI/reporting layer. |
| `learning/` | Provides a common learner trait plus Baum–Welch and AMaLGaM-backed implementations. |
| `initialization/` | Builds initial model parameters using K-means, random, sparse, hinted, or balanced strategies. |
| `number_states_finder/` | Searches a configured state-count range using clustering or Bayesian Information Criterion (BIC)-penalized HMM fits. |
| `amalgam_integration/`, `amalgam_tools.rs` | Encode HMM parameters as optimization variables and evaluate AMaLGaM candidates directly or through Baum–Welch refinement. |
| `hmm_tools.rs` | Supplies state-indexed one-, two-, and three-dimensional matrix helpers. |

### `src/optimization/` — reusable optimizer

The HMM AMaLGaM learner is built on a more general optimization layer:

- `optimizer.rs` defines common optimizer, fitness-function, and fitness-result traits.
- `amalgam_idea.rs` and `amalgam_idea/` implement the AMaLGaM-style estimation-of-distribution algorithm, its parameters, progress reporting, and tests.
- `multivariate_gaussian.rs` implements stable Gaussian sampling and covariance handling.
- `variable_subsets.rs` and `set_of_var_subsets.rs` split dependent variables into subsets and reconstruct full candidate solutions.
- `constraints.rs` describes bounds and other constraints applied to sampled variables.
- `tools.rs` contains shared selection helpers.

Keeping this machinery independent of the HMM representation makes it usable for other continuous optimization problems.

## Signal-analysis pipeline

The HMM is assembled in this fixed order:

```text
Number of States Finder → Initializer → Learner → Analyzer
       optional            optional     optional    always present
```

Execution flows from the first enabled component on the left toward the Analyzer. Adding more components removes more manual input:

| First enabled component | Input requested when analysis starts | What is automated |
| --- | --- | --- |
| Analyzer | States, state noises, start matrix, transition matrix | Only final inference |
| Learner | Initial parameters (the exact form depends on the learner) | Parameter fitting |
| Initializer | Number of states | Initial parameters and fitting |
| Number of States Finder | No model parameter input; the preprocessed sequences are sufficient | State count, initialization, fitting, and inference |

### Analyzer

The Analyzer validates a complete HMM and performs final inference. State means must be unique, state noises must be positive, the start probabilities must sum to one, and each transition-matrix row must sum to one. The GUI can reset matrices to balanced values or normalize entered rows.

### Learner

- **Baum–Welch** uses expectation maximization to improve an initial HMM. Its stopping rule can use a maximum iteration count, one-step convergence, or plateau convergence with relative or absolute likelihood changes.
- **AMaLGaM** explores distributions over HMM parameters. It supports several dependency layouts, direct likelihood evaluation or Baum–Welch-assisted fitness, iteration memory, and a configurable iteration limit. It is generally more exploratory, but also more computationally expensive.

### Initializer

The Initializer generates parameters before learning:

- State values: one-dimensional K-means, random values, sparse values across the observed range, or user-provided state hints.
- State noise: sparse initialization scaled by a configurable multiplier.
- Start and transition matrices: random or balanced probabilities.

State hints are useful when statistical optima disagree with experimentally expected states, but they are incompatible with automatic state-count finding.

### Number of States Finder

The state-count finder searches between configurable minimum and maximum values:

- **K-means clustering** compares clusterings using the silhouette or simplified silhouette score. It is fastest but least robust.
- **Baum–Welch** fits a standard predefined pipeline for each trial count and compares models using BIC.
- **Current setup** evaluates the configured Initializer and Learner for each trial count and also uses BIC. It offers the most flexibility and is usually the most expensive.

## Using the crate without the GUI

The GUI is only one consumer of the library. Code can construct an `HMM`, add and configure components, pass an `HMMInput`, run it, and retrieve results from the Analyzer. A compact example is kept in `src/main.rs` as `main_hmm`.

The typical programmatic flow is:

```rust
use fret_lib::signal_analysis::hmm::hmm_struct::{HMMInput, HMM};
use fret_lib::signal_analysis::hmm::learning::learner_trait::{
    LearnerSpecificSetup, LearnerType,
};
use fret_lib::signal_analysis::hmm::{InitializationMethods, StateValueInitMethod};

let sequences = vec![vec![0.2, 0.21, 0.75, 0.78]];

let mut hmm = HMM::new();
hmm.add_learner();
hmm.set_learner_type(LearnerType::BaumWelch)?;
hmm.setup_learner(LearnerSpecificSetup::BaumWelch {
    termination_criterion: None,
})?;

hmm.add_initializer()?;
let mut initialization = InitializationMethods::new();
initialization.state_values_method = Some(StateValueInitMethod::StateHints {
    state_values: vec![0.2, 0.75],
});
hmm.set_initialization_method(initialization)?;

hmm.run(HMMInput::Initializer {
    num_states: 2,
    sequence_set: sequences,
})?;

let analyzer = hmm.get_analyzer();
println!("states: {:?}", analyzer.get_states()?);
```

In an application, return or otherwise handle the concrete error type instead of using the snippet's `?` placeholders without a surrounding `Result`.

## Example use case

This example walks through the full FRET-Lib workflow using the files in [`traces_data_with_intruder/`](traces_data_with_intruder/). The directory contains two valid iSMS traces—one with donor photobleaching and one with acceptor photobleaching—as well as intentionally invalid inputs that demonstrate the loader's validation behavior.

To demonstrate the complete automated pipeline, the first analysis is performed without assuming the number or approximate values of the FRET states. The automated result is then compared with a second run guided by observations from the raw data.

### Step 0 — Main tab

Launch the application with `cargo run --release`. FRET-Lib initially opens on the **Main** tab, which summarizes the processing pipeline:

- Trace loading is available at the top.
- Preprocessing and filtering controls are on the left.
- Signal-analysis components are configured on the right.
- Status messages and errors appear in the console panel at the bottom.

![Initial FRET-Lib Main tab](<figures/Main tab.png>)

> **Important:** changes made in any settings window are not stored until the **Set** button is pressed.

### Step 1 — Load traces

Click **+ Load Traces** to open the loader. Individual files can be selected with **Load File**, while **Load Directory** attempts to load every file in a selected directory.

![Trace-loading window before selecting data](<figures/Load trace clean.png>)

For this example, choose **Load Directory** and select `traces_data_with_intruder/`. When the loader encounters an invalid file, it displays the reason for the failure. Valid files from the same directory are retained and listed in the window, so a single bad input does not discard the rest of the dataset.

![Trace-loading window after loading the example directory](<figures/Load traces after.png>)

Once at least one valid trace has been loaded, the **Raw Analysis** tab appears. Use the trace selector above the plot to switch between FRET pairs. The following plots show the original, unprocessed donor-photobleaching and acceptor-photobleaching traces.

![Raw FRET trace containing donor photobleaching](<figures/Raw fret D bleach plot.png>)

![Raw FRET trace containing acceptor photobleaching](<figures/Raw fret A bleach plot.png>)

The histogram combines FRET-efficiency samples from every loaded trace. At this stage it also includes post-photobleaching values, which can obscure the distribution relevant to the molecular states.

![Histogram of all raw FRET values](<figures/Raw histogram.png>)

### Step 2 — Preprocess the traces

Filtering remains enabled for this example so that photobleaching detection is performed, but no trace-rejection tests are activated. As a result, both valid traces pass preprocessing. Click **Run Preprocessing** and confirm the result in the console:

```text
Preprocessing completed successfully.
No points rejected.
```

The preprocessing stage truncates each trace at its first detected photobleaching event. The updated plots therefore retain the useful FRET segment and remove the post-bleach tail.

![Preprocessed donor-photobleaching FRET trace](<figures/D bleach fret plot.png>)

![Preprocessed acceptor-photobleaching FRET trace](<figures/A bleach fret plot.png>)

The combined histogram is updated at the same time. In this example, visual inspection suggests approximately four FRET states around `0.35`, `0.50`, `0.80`, and `0.90`. We will initially ignore that observation to test the fully automated workflow, then use it later to guide a second analysis.

![Histogram after photobleaching detection and preprocessing](<figures/Histogram.png>)

### Step 3 — Optional manual Analyzer input

The Analyzer is always present at the end of the signal-analysis pipeline. If all optional components are disabled, pressing **Run Signal Analysis** opens a form in which the entire HMM can be entered manually.

State values and noises are entered as lists. The start matrix uses a one-dimensional Python-style list, while the transition matrix uses a two-dimensional list of lists. **Reset Matrix** creates balanced probabilities and **Correct Matrix** normalizes the entered values.

![Manual Analyzer input window](<figures/Analyzer input.png>)

This example will use the automated components instead, so no manual Analyzer values are required.

### Step 4 — Configure the Learner

The Learner estimates the HMM parameters from the preprocessed sequences. Select **AMaLGaM Idea**, which is more robust to poor initial parameters than Baum–Welch alone. Keep the default configuration:

- **All Independent** dependency structure
- **Direct** fitness evaluation
- Iteration memory enabled
- The default maximum number of iterations

![Default AMaLGaM Learner settings](<figures/Amalgam Settings.png>)

Press **Set** to store the Learner configuration.

### Step 5 — Configure the Initializer

The Initializer supplies the Learner with an initial HMM. Keep the default strategies for the first run. The settings window separates initialization into four tabs: state values, state noise, the start matrix, and the transition matrix.

![Default state-value initialization settings](<figures/State Values settings.png>)

![Default state-noise initialization settings](<figures/State Noise settings.png>)

![Default start-matrix initialization settings](<figures/Start Matrix settings.png>)

![Default transition-matrix initialization settings](<figures/Transition Matrix settings.png>)

Press **Set** after reviewing the settings.

### Step 6 — Configure the Number of States Finder

The default pipeline includes the Learner and Initializer but not the Number of States Finder. Click the **+** button next to the component to enable it. The complete pipeline should now be visible on the Main tab.

![Main tab with the complete automated pipeline enabled](<figures/Main tab full.png>)

Open the component's settings and select **Current Setup**. This strategy tests candidate state counts using the Learner and Initializer configured in the previous steps. It is more computationally expensive than the K-means or fixed Baum–Welch strategies, but it preserves the flexibility of the selected pipeline. Set the desired minimum and maximum state counts, then press **Set**.

![Number of States Finder settings](<figures/NSF settings.png>)

### Step 7 — Run and inspect the automated analysis

Click **Run Signal Analysis**. Since the Number of States Finder is the first enabled component, the preprocessed sequence set is sufficient input and no state count is requested. This configuration can take some time because a complete HMM pipeline is evaluated for multiple candidate state counts.

After a successful run, open the **Analysis** tab. The fully automated pipeline selects a two-state model for this dataset.

![Overview of the fully automated two-state analysis](<figures/Bad analysis.png>)

The fitted state distributions and occupancy plot show that one state dominates the dataset.

![States selected by the fully automated analysis](<figures/Bad result states.png>)

![State occupancy from the fully automated analysis](<figures/Bad state occupancy.png>)

The idealized sequence can be inspected over each original observation trace:

![Automated idealized sequence over the donor-photobleaching trace](<figures/D bleach bad analysis.png>)

![Automated idealized sequence over the acceptor-photobleaching trace](<figures/A bleach bad analysis.png>)

Although this is the statistical optimum found by the configured search, it merges the two higher states and misses the lower state visible in the acceptor-photobleaching trace. This illustrates an important limitation of automatic state-count selection when state occupancies are strongly unbalanced.

### Step 8 — Apply manual state hints

To guide the model toward the four states suggested by the preprocessed data:

1. Disable the **Number of States Finder** using the **x** control.
2. Open the Initializer settings.
3. Change state-value initialization to **State Hints**.
4. Enter the approximate values `[0.35, 0.5, 0.8, 0.9]`.
5. Press **Set**, then run the signal analysis again.

The number of hints fixes the expected state count, so no separate state-count input is needed.

![Initializer configured with four state-value hints](<figures/State hints try settings.png>)

The updated analysis now identifies all four expected FRET states and reports their parameters, occupancies, and probability matrices.

![Analysis window after applying state hints](<figures/Good analysis window.png>)

The new idealized sequences also align more closely with the visible levels in both experimental traces.

![Improved idealized sequence over the donor-photobleaching trace](<figures/D bleach good plot.png>)

![Improved idealized sequence over the acceptor-photobleaching trace](<figures/A bleach good plot.png>)

### Step 9 — Export and discuss the result

Use the export control in the **Analysis** tab to save a structured text report containing the HMM setup and the inferred state sequences. An example of the generated format is available at [`hmm_setup_report.txt/hmm_setup_and_sequences_report.txt`](hmm_setup_report.txt/hmm_setup_and_sequences_report.txt).

This example demonstrates both sides of FRET-Lib's design. The automated components can build and evaluate an entire HMM pipeline with little prior knowledge, but a statistically preferred model is not always the most scientifically meaningful one. Retaining manual controls—particularly state hints—allows experimental knowledge to guide the optimizer when state populations are uneven or weakly represented.

---

FRET-Lib was developed in the Caneva Lab at Delft University of Technology as an in-house, adaptable smFRET analysis tool.
