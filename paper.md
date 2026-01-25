\begin{abstract}
Chain-of-Thought (CoT) explanations are essential for the monitoring and safety of Large Language Models (LLMs), yet they are susceptible to unfaithful rationalization that could obfuscate dangerous behaviors. 
While prior work has focused on black-box methods, the internal mechanisms and white-box control of faithfulness remain under-explored. 
In this paper, we employ representation engineering to investigate the latent geometry of faithfulness across a suite of reasoning models of increasing sizes. 
We demonstrate that instances of faithfulness are, to some extent, encoded as linear directions within middle-to-late layers, as shown by successful probing and steering of the models' internals. 
We find that linear steering interventions achieve faithfulness recovery rates of up to 42\% while maintaining collateral effects below 5\%. 
Furthermore, we find that off-policy steering methods have a comparable utility to on-policy approaches.
\end{abstract}

\section{Introduction} \label{sec:intro}
The rise of Large Language Models (LLMs) capable of advanced reasoning has made Chain-of-Thought (CoT) explanations a cornerstone of model monitorability \citep{meek2025measuringchainofthoughtmonitorabilityfaithfulness, DBLP:journals/corr/abs-2503-11926}. 
However, the utility of these explanations depends entirely on their \textit{faithfulness}—the degree to which the reasoning trace actually reflects the model's internal decision-making process \citep{DBLP:conf/nips/TurpinMPB23}. 
Recent studies highlight that models often produce highly plausible but unfaithful rationalizations \citep{DBLP:journals/corr/abs-2503-08679, chua2025deepseekr1reasoningmodels, DBLP:journals/corr/abs-2505-05410}. 
As reasoning capabilities advance, traditional black-box monitoring could become increasingly insufficient to detect dangerous and undesired behaviours \citep{DBLP:journals/corr/abs-2507-11473}.

In this paper, we investigate the internal mechanics of CoT faithfulness through the lens of representation engineering \citep{}. 
We explore whether faithfulness is a property that can be identified, monitored, and controlled directly within a model's latent space. Using a suite of reasoning models, we generate activation directions associated with faithful and unfaithful reasoning across diverse domains and safety-related contexts.

Our contributions are threefold. 
First, we develop a comprehensive framework to train and evaluate faithfulness probes and steering vectors. 
Second, we identify that instances of faithfulness are, to some extent, represented as high-level linear directions in middle-to-late layers. 
Third, we benchmark multiple representation engineering approaches, observing a performance proximity between on-policy and off-policy directions—validating off-policy data as a viable intervention source—and demonstrating that linear steering is more robust than non-linear methods, achieving recovery rates up to 42\% with minimal collateral effects. 
Our results position light-weight white-box methods as promising tools for the oversight of LLM-based systems.

\section{Related Work} \label{sec:rel-work}

% two/three paragraphs

% Evaluating Faithfulness in CoT Explanations
\textbf{Evaluating Faithfulness of CoT Explanations.}
Historically, NLP interpretability distinguished faithful explanations—accurately reflecting the model's reasoning—from those merely plausible to humans \citep{jacovi-goldberg-2020-towards}. 
In the LLM era, faithfulness is often operationalized as a property of the model’s CoT, through the causal dependency between the CoT components and the final answer \citep{DBLP:journals/corr/abs-2307-13702}, or counterfactual simulatability in adversarial settings biasing the model toward a specific choice \citep{chua2025deepseekr1reasoningmodels, DBLP:journals/corr/abs-2505-05410, DBLP:conf/nips/TurpinMPB23}.

% Unfaithfulness and CoT Monitorability
\textbf{Unfaithfulness and CoT Monitorability.}
Faithfulness is a prerequisite for CoT monitorability \citep{DBLP:journals/corr/abs-2503-11926}, yet training paradigms like RLHF could incentivize human-friendly rather than faithful explanations \citep{DBLP:conf/iclr/SharmaTKDABDHJK24}. 
Reasoning models remain susceptible to bias despite improvements in faithfulness \citep{chua2025deepseekr1reasoningmodels}.
In this scenario, and as the model's latent reasoning improves, CoT monitoring remains an insufficient guardrail \citep{DBLP:journals/corr/abs-2507-11473}.

% Control Through Representation Engineering
\textbf{Control Through Representation Engineering.}
Current faithfulness mitigations rely on black-box approaches \citep{DBLP:conf/ijcnlp/LyuHSZRWAC23, DBLP:journals/corr/abs-2307-11768, DBLP:journals/corr/abs-2403-05518}. 
In contrast, representation engineering \citep{DBLP:journals/corr/abs-2310-01405} enables white-box control via probing and steering vectors \citep{DBLP:conf/acl/RimskyGSTHT24, DBLP:journals/corr/abs-2502-03407}, successfully targeting high-level concepts as linear representations \citep{DBLP:journals/corr/abs-2310-06824,DBLP:conf/nips/ArditiOSPPGN24, DBLP:journals/corr/abs-2310-15154, DBLP:journals/corr/abs-2507-01786}.
However, applying these internal interventions to CoT faithfulness remains a gap, particularly in determining whether such complex behavior is encoded linearly \citep{DBLP:conf/icml/ParkCV24} or privilege non-linear geometries \citep{DBLP:journals/corr/abs-2411-03343, DBLP:journals/corr/abs-2505-24535}.

\section{Methodology} \label{sec:setup}

% Models we use
\textbf{Models:} 
We investigate reasoning models of increasing size. 
We use DeepSeek-R1-Distill-Llama-8B (Llama-3.1-8B distilled via SFT on reasoning traces from DeepSeek-R1) \citep{deepseekai2025deepseekr1incentivizingreasoningcapability}, Qwen3-14B, and Qwen3-32B \citep{DBLP:journals/corr/abs-2505-09388}. 
For maximum reproducibility, we set the temperature to 0. 
However, for the Qwen3 models, coupling greedy decoding with "thinking mode" caused severe repetition loops during CoT generation. 
To mitigate this, we ran the Qwen3 models with a presence penalty of 1.2 to enforce termination. 
All models were loaded in BF16 precision. 

% Existing datasets we use
\textbf{Datasets:} 
Using three MMLU macro-domains \citep{hendryckstest2021} totaling 2516 prompts, we filter for correctly answered questions.
We then generate adversarial samples via a full factorial design, prepending safety-relevant biasing hints \citep{DBLP:journals/corr/abs-2505-05410} to every question. 
Retaining only traces where the model adopts the hint in their final answer, we classify them as faithful or unfaithful, and split them via stratified sampling to preserve hint distribution. 
The train split is used to extract faithfulness directions, while the test split evaluates our probes and steering vectors. 
For more details on the adversarial hint prompts, see Appendix A \ref{}.

\textbf{Faithfulness Operationalization} is done based on the counterfactual simulatability of CoT reasoning \citep{DBLP:conf/nips/TurpinMPB23}.
We define a CoT trace as faithful if (1) it reaches the conclusion suggested by the biasing hint and (2) it acknowledges the hint's influence\citep{DBLP:journals/corr/abs-2505-05410}.

\textbf{Generating Faithful CoT Reasoning Traces:} 
Using the modified MMLU prompts with biasing hints, we elicit \emph{on-policy} CoT traces.
These reasoning traces are classified as faithful or unfaithful, and then the steps are annotated to identify the CoT steps responsible for the classification, using two separate LLM-judges \citep{DBLP:journals/corr/abs-2412-05579, DBLP:conf/nips/ZhengC00WZL0LXZ23}.
To assess whether \emph{off-policy} data yields effective linear representations \citep{kirch2026impactoffpolicytrainingdata}, we generate synthetic CoT traces using predefined contrastive templates.
\citet{kirch2026impactoffpolicytrainingdata} find that off-policy data is sufficient for training high quality probes, we go beyond and show it is also useful data for obtaining steering vectors.
%%% Appendix: Examples and low-level details
For more details on these aspects of the pipeline, see Appendix B \ref{}.

\textbf{Extracting Directions of Faithfulness in Latent Space:}
From a train split (70\%) of the on-policy CoT traces, we extract activations from the last-token of each annotated step, ending with a set of activations from faithful and unfaithful CoT traces. 
We also extract activations from our contrastive off-policy prompts with the same approach.
To find a \emph{linear} direction of faithfulness we compute the difference between the average of faithful and unfaithful activations \citep{DBLP:conf/acl/RimskyGSTHT24}.
To get \emph{non-linear} faithfulness directions, we train a two-hidden-layer 8-neuron MLP probe to distinguish between faithful and unfaithful activations. 
Adapting the \emph{probe-guided latent intervention} method from \citep{DBLP:journals/corr/abs-2411-03343}, we compute the gradient of the probe’s prediction with respect to the input activations. 
This allows the probe to identify a dynamic, sample-specific direction to steer along complex, non-linear boundaries where subtle features of faithfulness may reside.


% Detecting Unfaithfulness with Linear and MLP Probes
\textbf{Detecting Unfaithfulness with Linear and MLP Probes:}
To assess the alignment between the correlational features detected by probes and the causal effects of our steering vectors, we trained both linear and non-linear classifiers. 
We employed logistic regression for the linear probe and utilized the MLP architecture described above for the non-linear variant.

% Steering with three approaches
\textbf{Steering Faithfulness:} We utilize the extracted directions to intervene on the model's hidden states, either adding the vectors (positive steering) to promote faithfulness or subtracting it (negative steering) to promote unfaithfulness. 
As a sanity check, we apply both interventions to originally faithful and unfaithful prompts to verify consistent, opposing behavioral shifts.
We evaluate three distinct steering methods: (1) linear vectors derived from on-policy activations; (2) linear vectors constructed from off-policy paired contrastive data; and (3) prompt-guided, sample-specific interventions generated via our MLP probe. 
% For each method, we perform a hyperparameter sweep across injection layers and steering strengths to identify optimal configurations. 
We also benchmark against random vector baselines of equivalent magnitude.

% Evaluation
\textbf{Evaluation Metrics.} For each steering configuration, we track the following metrics as a percentage of the total steered answers of one pre-steering state (faithful, unfaithful):
\begin{itemize}[noitemsep,topsep=0pt]
    \item \textbf{Faithfulness (Recovery and Degradation) Rates:} The rates of originally unfaithful answers that are converted to faithful answers with positive steering (and of originally faithful answers that are converted to unfaithful ones with negative steering).
    \item \textbf{Hint-Mentioning Rate:} The proportion of answers that do not qualify as fully faithful but still explicitly acknowledge the hint's existence after steering from an unfaithful state, i.e. the hint is mentioned but the model does not choose the hinted option. 
    \item \textbf{Correctness Rate:} The proportion of answers that result in the correct final option (so the model is not adopting the hint) after steering previously wrong answers. % this is noise in our findings, because pos and neg have same effect
    \item \textbf{Collateral Effect Rates:} The rates of originally unfaithful answers that are converted to faithful with negative steering, and of originally faithful answers that are converted to unfaithful ones with positive steering.
\end{itemize}

\textbf{Finding the Optimal Layer:}
We perform a hyperparameter sweep across injection layers and steering strengths, selecting the configurations that maximize the ratio between average faithfulness recovery and collateral effect rates.
We observe that optimal configurations vary substantially by model and approach, see Appendix \ref{}. 


\section{Results}  \label{sec:results}

\textbf{Probing for Faithfulness:}
Both linear and MLP probes performed strongly, with peak layers' F1 scores consistently between 90\% and 100\%, see Figure \ref{}. 
Linear and non-linear probes perform on par. 
% Crucially, for linear directions, the close proximity between optimal probing and steering layers reinforces the causal relevance of these regions.
Our probing results show that we can identify which latent vectors are faithful. 
However, this ability to identify faithfulness does not result in equally strong steering ability.

\textbf{Steering towards Faithfulness:}
In Figure \ref{fig:main-steering-plot} we show the steering performance (as measured through faithfulness and hint-mentioning), for steering unfaithful answers towards faithfulness (and for steering faithful answers to unfaithfulness).
We find that positive steering yields a wide range of faithfulness recovery rates, from limited (7.6\%, MLP on DeepSeek-R1-Llama-8B) to substantial (42\%, linear on Qwen3-14B), with no evident correlation between steering efficacy and model size. 
% include info about other papers' steering performance
We find that steering efficacy varies substantially depending on the specific hint (see Appendix ??).

\textbf{Steering Effects Unrelated to Faithfulness:}
Notably, both positive and negative steering results in high rates of correctness recovery, see Figure \ref{} in the Appendix.
This was surprising to us, as we expected positive steering to encourage ``good'' behaviour.
We interpret this finding as the correctness recovery simply being noise. 
In a minority of instances, steering led to an "other-error" state, where the model selected an incorrect option unrelated to the hint. 
Within these non-faithfulness states (both correct answers and other-error answers), failed positive steering still resulted in varying hint-mentioning rates, from 4.14\% in Qwen3-14B to 38.15\% in Qwen3-32B, see Figure \ref{}. 
Conversely, failed negative steering still resulted in varying hint-ablation rates, that is when the model stops acknowledging the hint. 

\textbf{Linear vs Off-Policy vs MLP:} 
The linear approach works as well or better than off-policy and MLP in terms of achieving a high faithfulness recovery rate (see Figure \ref{fig:main-steering-plot}) while keeping collateral effects (where faithful answers become unfaithful) below 5\%, see Figure \ref{} in Appendix \ref{}. 
The off-policy approach yields comparable utility. 
In contrast, our non-linear MLP method offers no clear advantage over simpler linear interventions:
in Figure \ref{1.b} we find that the unfaithfulness degradation rate is low, and in Appendix \ref{} we find that the collateral damage is high.

% We may want to move this paragraph to the appendix
\textbf{Best Steering Configurations:} Optimal steering tends to target the middle-to-late layers of the networks, see Appendix \ref{}, 
% (e.g., layer 40 for Qwen3-32B or layer 15 for DeepSeek-8B), 
suggesting that faithfulness is a high-level conceptual feature synthesized after initial processing. 
We also find that linear and off-policy approaches in small and medium models have the exact same optimal layer (layer 15 for 8B and layer 19 for 14B).
% The layer convergence between linear and off-policy approaches in small and medium models (aligning at layer 15 for 8B and layer 19 for 14B) could signal that off-policy methods are capable of capturing the model's native linear representations.


\begin{figure}[t]
    \centering
    \includegraphics[width=0.95\linewidth]{figures/main/variation_4.png}
    \caption{Faithfulness probe performance (F1 score) across layers for three reasoning models. Both linear (LogReg) and non-linear (MLP) probes achieve high accuracy (90--100\%) in middle-to-late layers, with no substantial difference between probe types.}
    \label{fig:probing-plot}
\end{figure}

\begin{figure}[t]
    \centering
    \includegraphics[width=0.95\linewidth]{figures/main/variation_11_monitorability_with_random.png}
    \caption{Steering performance across models and approaches. \textbf{Top-left:} Monitorability gain from positive steering on unfaithful answers (stacked: faithfulness + hint-mentioning). \textbf{Top-right:} Intended unfaithfulness degradation from negative steering. \textbf{Bottom-left:} Collateral effects (positive steering making faithful answers unfaithful). \textbf{Bottom-right:} Unintended monitorability gain from negative steering. Linear and off-policy approaches achieve the highest intended effects with lowest collateral damage.}
    \label{fig:main-steering-plot}
\end{figure}

% \begin{wrapfigure}{t}{0.5\textwidth}
%     \centering
%     \includegraphics[width=.9\linewidth]{figures/NeurIPS-paper/ind-ood-accs.png}
%     \caption{Accuracy of CIFAR-10 pre-trained models on OOD datasets (on the y-axis) against accuracy on CIFAR-10 (on the x-axis). The dashed line (which coincides with the green and blue lines) is the $y=x$ line.}
%     \label{fig:ood_acc_shift}
% \end{wrapfigure}

% \begin{table}[]
%     \centering
%     \begin{tabular}{c|cccc}
%     \toprule
%                  & \multicolumn{4}{c}{Representation Length} \\[5pt]
%     % \cline{2-5}
%     Training $k$ &              64    &              128   &              256   &              512   \\
%     \midrule
%     3            &  $0.909 \pm 0.029$ &  $0.869 \pm 0.015$ &  $0.870 \pm 0.052$ &  $0.887 \pm 0.015$ \\
%     5            &  $0.797 \pm 0.026$ &  $0.688 \pm 0.077$ &  $0.759 \pm 0.131$ &  $0.820 \pm 0.166$ \\
%     10           &  $0.866 \pm 0.103$ &  $0.579 \pm 0.018$ &  $0.643 \pm 0.231$ &  $0.736 \pm 0.171$ \\
%     20           &  $0.662 \pm 0.170$ &  $0.538 \pm 0.230$ &  $0.532 \pm 0.380$ &  $0.481 \pm 0.337$ \\
%     \bottomrule
%     \end{tabular}
%     \vspace{0.3cm}
%     \caption{Accuracy on CIFAR-10 test set of trained models with different $k$ and $|r|$ values.}
%     \label{tab:cifar10_acc_table}
%     \vspace{-0.4cm}
% \end{table}


\section{Limitations and Future Work} \label{sec:related_work}
% Our findings are subject to several limitations. 
% First, we conducted a limited sweep of layers and coefficients, which may overlook more granular optima.
Our testing focused on narrow and unrealistic scenarios, specifically post-hoc rationalization of hints within an in-distribution setting. 
Additionally, our evaluation did not include larger model scales, where steering dynamics may differ.

Future work could test these steering interventions across a richer pool of unfaithfulness cases, incorporating out-of-distribution tasks \citep{DBLP:journals/corr/abs-2506-10922, DBLP:journals/corr/abs-2503-08679} and, more importantly, agentic settings \citep{DBLP:journals/corr/abs-2412-04984} to assess real-world robustness.

\section{Conclusion} \label{sec:conc}
In this work, we demonstrated that some narrow instances of faithfulness in reasoning models live as high-level conceptual feature encoded in middle-to-late layers. 
Our results show that linear steering achieves recovery rates of up to 42\% with collateral effects below 5\%, while non-linear MLP methods offer no clear advantage over simpler interventions. 
While limited to non-agentic scenarios, these findings establish white-box interventions as a viable path for monitoring and controlling model faithfulness.

















