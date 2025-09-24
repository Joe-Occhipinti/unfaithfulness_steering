# Project Summary

This research project aims to investigate Chain-of-Thought Unfaithfulness in reasoning models. After eliciting faithful and unfaithful behaviours with biasing hints in the prompts (similar to Turpin et al., 2023), we want to test if the acts of being faithful and unfaithful are linearly encoded in the model's activations. We want to test if we can isolate a "direction towards faithfulness" and compute a steering vector out of it using methods like contrastive activation addition (similar to Rimsky et al., 2024).

# Workflow
We need to build the main scripts and sub modules to recreate this workflow. In order to test the entire pipeline end to end while we build it, we will use small amounts of syntethic data with the right jsonl schemes. 

1. Baseline run: get baseline performance on MCQs loaded from MMLU (2-3 domains) --> main script is a colab notebook cloning the repo from github and importing scripts
     └→ input data: load N MMLU subjects, create baseline input prompts from the MMLU MCQs.
     └→ model: load the model with the setup to generate text performing multiple forward passes.
     └→ generate: get the model performance generating text continuing the baseline input prompts.
     └→ eval: validate response format following, extract answer letter, compute accuracy, label them (correct, wrong)
     └→ output data: store question, baseline input prompts + baseline generated text (= baseline output prompt) + answer letter + ground truth letter (converting from number to letters like in 0->A) + accuracy labels (correct, wrong).

2. Hinted run: get performance on baseline correct answers, but now the prompts have a hint. --> main script is a colab notebook cloning the repo from github and importing scripts
     └→ data: load baseline input prompts that were labeled as correct in the baseline run, modify them adding a hint at the start of the prompt to create biased input prompts.
     └→ model: load the model with the setup to generate text performing multiple forward passes.
     └→ generate: get the model performance generating text continuing the biased input prompts.
     └→ eval: validate response format following, extract answer letter, compute accuracy, label them (correct, wrong), label them also as biased if wrong, not-biased if correct.
     └→ intermediate output data in hinted_[date]: store the hint verbatim, the hinted input prompts, the hinted generated texts, the merged of the two = biased prompts, answer letter, ground truth letter, hinted letter, accuracy labels (correct, wrong), bias labels ("biased" if wrong, "not-biased" if still correct).
     └→ eval faithfulness: Annotate and classify for faithfulness biased prompts with a LLM-annotator + rule table to classify for faithfulness/unfaithfulness/other global labels of the CoT.
        └→ input data: load biased prompts (hinted input prompts + hinted generated texts) from the intermediate output of the hinted_eval.py run
        └→ model: set up a proprietary model API and parameters. gemini 2.5 pro
        └→ annotate: call the model with a annotatation prompt in the system prompt and the biased prompt to annotate in the user query. Get the annotated text from the called model output.
        └→ classify: using a rule-table that processes the annotated biase prompt, and looking at the combination of tags, answer letter, hinted letter, label each annotated biased prompt with the corresponding resulting label
        └→ eval at the prompt-level: count and save distribution across rule-table labels. 
        └→ eval at the reasoning step level: count and save distribution across labels (per prompt)
        └→ output data in annotated biased [date]: store input hinted prompts that were labeled as biased not annotated, corresponding hinted generated texts not annotated, biased prompts non annotated (hinted input + hinted generated output), annotated biased prompts, with their answer letter, hinted letter, the ground truth letter, their prompt-level classification ("faithful", "unfaithful", other labels). This dataset must be divided in train, split, val.

3. Extract hidden state activations from annotated biased prompts at the level of specific locations marked by closing tags (look at legacy code to how to do that), in prompts in annotated biased [date] that were labeled as faithful and unfaithful, and store them in a dataset of activations. The dataset must still be divided in train, val, test respecting the splits from the annotated biased prompts dataset (so, maintaining a prompt-wise hierarchy: for every prompt --> for every layer --> for every label --> every activations) --> main script is a colab notebook cloning the repo from github and importing scripts
    └→ input data: load annoated biased prompts
    └→ model: load model to perform one single forward pass (not generate text).
    └→ extraction loop: extract activations at the level of specific locations of selectable tags in the annotated prompts.
    └→ store activations in dataset of activations splitted in train val test, maintaining a prompt-based organization.
    This function was well implemented in the legacy code so look at it. just extracting parts that can constitute reusable modules to put in src.
    
4. separability_analysis: basically a main script aggregating some core logic behind testing separability, so the correlation with linearly encodedness (not causation, that is steering). In the main script, the positive and negative activations are tunable parameters, and basically we should put in these variables the labels from the annotation that we want to consider positive and negative for that run of analysis. The same thing will apply to computing steering vectors.
     └→ compute cosine similarity distribution across layer of specific positive and negative activations labels from the dataset of activations. Where positive and negative are tunable parameters corresponding to the set of labels from the dataset we group in each category. (e.g., positive = F vs negative = U, positive = F_final vs. negative = U_final, positive = F + F_final vs. negative = U + U_final, positive = F + F_wk vs negative = U etc.)
     Same thing but for norm changes distribution.
     └→ training and testing linear probes with linear regression for every layer, to classify different combinations of positive and negative activations from the activation dataset. plotting layer-wise the accuracy or AUC. Where positive and negative are tunable parameters based on the set of labels from the activation dataset we group in each category. Must be trained on train + val splits, tested on test split? Do we need the val set for something in this case? I don't know.
     
5. compute steering vectors from the mean difference between positive and negative activations, where positive and negative are tunable parameters based on the set of labels from the activation dataset we group in each category. (e.g., F vs U, F_final vs. U_final, F + F_final vs. U + U_final, F + F_wk vs U etc.)

9. tune_steering_vectors --> main script is a colab notebook cloning the repo from github and importing scripts
    └→ input data: load hinted input prompts not annotated from the annotated biased prompt dataset, val split.
    └→ model: load model to generate text and be steered during inference with activation addition.
    └→ vectors: load the steering vectors computed from all layers.
    └→ steer: run the model over the hinted input prompts, adding the steering vector at each forward pass to the last token. Add each steering vector to the corresponding source layer. Sweep many coefficients (like from 0.5 to 1, and -0.5 to -1)
    └→ eval: validate response format following, extract answer letter.
    └→ eval faithfulness entire pipeline (from annotation, to classifation, to visualization) on steered prompts (hinted input prompts + steered generated output). This should show us which combination of layer-coefficient is the best to turn unfaithful prompts to faithful.
    └→ output: save hinted input prompt, unsteered biased prompt, steered biased prompt, unsteered answerletter, hinted letter, ground truth, steered answer letter, old faithfulness classification, new faithfulness classification

10. test_final_performance --> main script is a colab notebook cloning the repo from github and importing scripts
    └→ input data: load hinted input prompts not annotated from the annotated biased prompt dataset, test split.
    └→ model: load model to generate text and be steered during inference with activation addition.
    └→ vectors: load the steering vector corresponding to the best performing layer (tunable parameter) Also, the coefficient is tunable and it could correspond to the best performing coefficients.
    └→ steer: run the model over the hinted input prompts, adding the steering vector at each forward pass to the last token.
    └→ eval: validate response format following, extract answer letter.
    └→ eval faithfulness entire pipeline (from annotation, to classifation, to visualization) on steered prompts (hinted input prompts + steered generated output). This should show us if the selected steering vector and coefficient are truly good and not overfitting.
    └→ output: save hinted input prompt, unsteered biased prompt, steered biased prompt, unsteered answerletter, hinted letter, ground truth, steered answer letter, old faithfulness classification, new faithfulness classification.


# Repository Structure

workflow_notebooks (we will write them as py files for now)
    baseline_eval.py
    hinted_eval.py
    activations.py
    separability.py
    steering_vectors.py
    tune_steering_vectors.py
    test_steering_vectors.py
src
    model.py
    config.py
    data.py
    performance_eval.py
    faithfulness_eval.py
    separability_analysis.py
    extract_activations.py
    prompts.py
    plots.py
    steering.py

prompts
    faithfulness_steps_annotator.txt

data
    behavioural
        baseline_[DD-MM-YYYY].jsonl
        hinted_[DD-MM-YYYY].jsonl
        steered_val_[DD-MM-YYYY].jsonl
        steered_test_[DD-MM-YYYY].jsonl

    annotated
        annotated_hinted_[date].jsonl
        annotated_steered_[date].jsonl

    activations
        activations_[source dataset]_[DD-MM-YYYY]
            prompt 1.pt
            ...
        ...

    datasets of activations
        activations_[source dataset]_[DD-MM-YYYY].pkl
        ...

    steering vectors
        steering_vector_[labels e.g., F_vs_U]_[DD-MM-YYYY].pkl
        ...

plots