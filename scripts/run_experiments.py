"""
Reproducible representation geometry benchmark & diagnostic probing experiment pipeline.

Evaluates:
- Multiple model families: Encoders (bert-base-uncased, roberta-base) and Decoders (gpt2, distilgpt2)
- Representation geometry metrics (Anisotropy, Uncentered SVD, MEV, Token Self-Similarity, CKA, Drift)
- Systematic 4-task probing suite (POS, NER, Topic, STS)
- Directly tests the research hypothesis:
  "Does geometric stabilization (SOD/CKA) correspond to functional linguistic usability?"

Results are saved to `benchmark_results/benchmark_data.json`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from repr_audit.auditor import RepresentationAuditor
from repr_audit.extractor import HiddenStateExtractor
from repr_audit.metrics import evaluate_sod_sensitivity
from repr_audit.probing import (
    compare_geometry_against_probing,
    evaluate_linear_classification_probe,
    evaluate_sts_similarity_probe,
)

# ---------------------------------------------------------------------------
# Standard linguistic evaluation benchmark data
# ---------------------------------------------------------------------------

BENCHMARK_SENTENCES = [
    # Topic 0: Finance & Banking
    "The central bank decided to lower interest rates to stimulate investment.",
    "Commercial banks reported significant quarterly earnings from retail mortgages.",
    "Inflationary pressures led the federal bank to tighten fiscal policies.",
    "She deposited her savings account checks at the local branch bank.",
    "Investment banking divisions experienced high merger and acquisition volume.",
    "The financial bank refused credit to high risk borrower applicants.",
    "Stock market equities rallied following the national bank announcement.",
    "Global banking regulations aim to prevent systemic liquidity crises.",
    "Assets held by commercial banks expanded during the economic boom.",
    "The bank issued credit lines to support small business operations.",

    # Topic 1: Geography & River Banks
    "The fisherman walked along the muddy bank of the winding river.",
    "Wildflowers and tall grass bloomed across the southern river bank.",
    "Heavy monsoon rains eroded the sandy bank near the village.",
    "We pitched our camping tents on the grassy bank overlooking the lake.",
    "A small wooden canoe was tied to an old tree on the river bank.",
    "The steep river bank made descending to the water dangerous.",
    "Birds nested quietly along the shaded bank of the stream.",
    "Melting spring snow caused the river to overflow its natural bank.",
    "Children threw flat stones from the gravel bank into the rushing current.",
    "Dense weeping willows lined both sides of the canal bank.",

    # Topic 2: Computing & AI
    "Deep neural networks learn hierarchical representations from large corpora.",
    "Transformer architectures use self attention mechanisms across token sequences.",
    "Optimization algorithms minimize loss functions using stochastic gradient descent.",
    "Language models predict next tokens through causal probability distributions.",
    "Linear probes measure linguistic information encoded in hidden activations.",
    "Distributed computing clusters train billion parameter foundation models.",
    "Vector embeddings capture syntactic and semantic regularities in geometry.",
    "Representation collapse occurs when contextual vectors become highly anisotropic.",
    "Modern benchmark evaluations measure few shot reasoning and generalization.",
    "Attention layers compute dot product affinities between queries and keys.",

    # Topic 3: Biology & Nature
    "Photosynthetic plants convert sunlight and carbon dioxide into sugars.",
    "Enzymes catalyze biochemical metabolic pathways within biological cells.",
    "Genetic mutations drive phenotypic variation through natural selection.",
    "Marine ecosystems support diverse marine species along coral reefs.",
    "Mammalian migration patterns correlate with seasonal temperature changes.",
    "Cellular respiration produces adenosine triphosphate in mitochondria.",
    "Endangered predators maintain trophic balance in temperate forest habitats.",
    "Avian species communicate territorial boundaries through acoustic song patterns.",
    "Microbial communities in soil recycle nitrogen and essential organic nutrients.",
    "Evolutionary biology traces phylogenetic lineages through genomic sequencing.",
]

TOPIC_LABELS = np.array([0] * 10 + [1] * 10 + [2] * 10 + [3] * 10)

# Target polysemous words in diverse contexts for true token-level self-similarity
POLYSEMOUS_TOKEN_SENTENCES = {
    "bank": [
        "The central bank lowered benchmark interest rates today.",
        "She deposited savings into her commercial bank account.",
        "He applied for a personal mortgage at the city bank.",
        "The robbers targeted an automated bank machine downtown.",
        "We sat peacefully on the grassy bank of the river.",
        "Floodwaters overflowed the steep river bank during the storm.",
        "A fisherman cast his line from the muddy bank of the stream.",
        "The airplane began to bank sharply toward the runway.",
    ],
    "crane": [
        "The tall construction crane lifted steel beams to the top floor.",
        "Operators navigated the heavy tower crane across the shipyard.",
        "The white whooping crane landed gracefully near the marsh.",
        "A flock of sandhill crane birds flew across the morning sky.",
        "He had to crane his neck to see the stage above the crowd.",
        "Pedestrians crane forward to catch a glimpse of the parade.",
    ],
    "apple": [
        "She ate a crisp red apple for her morning breakfast.",
        "The farmer harvested bushels of sweet apple fruit in autumn.",
        "Apple released a new operating system update for developers.",
        "Investors analyzed Apple stock performance after quarterly earnings.",
        "An apple orchard flourished in the fertile northern valley.",
        "Engineers at Apple designed custom silicon chips for smartphones.",
    ],
}

# POS Probing Sentences with target word & POS label
# (NOUN=0, VERB=1, ADJ=2, ADV=3)
POS_PROBING_DATA = [
    ("The scientist analyzed the complex experimental data carefully.", "scientist", 0),
    ("The engineer designed an efficient turbine for the project.", "engineer", 0),
    ("The doctor examined the patient with great compassion.", "doctor", 0),
    ("The river flowed smoothly through the wide green valley.", "river", 0),
    ("The computer processed millions of matrix operations.", "computer", 0),
    ("The musician composed an emotional acoustic melody.", "musician", 0),
    ("The committee decided to implement new environmental policies.", "committee", 0),
    ("The athlete trained intensely for the national marathon.", "athlete", 0),

    ("Researchers investigate new quantum computing phenomena.", "investigate", 1),
    ("Students study complex mathematics and algorithmic theory.", "study", 1),
    ("Technicians build robust electrical infrastructure worldwide.", "build", 1),
    ("Algorithms optimize multi objective performance constraints.", "optimize", 1),
    ("Teachers educate children with dedication and patience.", "educate", 1),
    ("Explorers discover ancient geological structures underground.", "discover", 1),
    ("Analysts forecast global economic recovery patterns.", "forecast", 1),
    ("Programmers debug unexpected software exceptions regularly.", "debug", 1),

    ("The team achieved remarkable scientific breakthrough results.", "remarkable", 2),
    ("The architecture exhibits elegant modular design principles.", "elegant", 2),
    ("The experiment yielded surprising anomalous measurements.", "surprising", 2),
    ("The company developed revolutionary therapeutic medications.", "revolutionary", 2),
    ("The algorithm demonstrates exceptional computational speed.", "exceptional", 2),
    ("The document contains comprehensive historical documentation.", "comprehensive", 2),
    ("The forest contains rare endangered botanical species.", "rare", 2),
    ("The solution presents severe practical engineering challenges.", "severe", 2),

    ("The car accelerated rapidly along the open highway.", "rapidly", 3),
    ("The algorithm converges smoothly toward the global minimum.", "smoothly", 3),
    ("The team worked diligently to finish the prototype.", "diligently", 3),
    ("Prices fluctuated wildly during market volatility.", "wildly", 3),
    ("The author wrote eloquently about philosophical traditions.", "eloquently", 3),
    ("The system operates reliably under extreme conditions.", "reliably", 3),
    ("The bird soared effortlessly above the ocean waves.", "effortlessly", 3),
    ("The patient recovered gradually following clinical treatment.", "gradually", 3),
]

# NER Probing Data (PER=0, LOC=1, ORG=2)
NER_PROBING_DATA = [
    ("Albert Einstein formulated the theory of general relativity.", "Albert Einstein", 0),
    ("Marie Curie conducted pioneering research on radioactive decay.", "Marie Curie", 0),
    ("Alan Turing laid the theoretical foundations of modern computing.", "Alan Turing", 0),
    ("Ada Lovelace wrote the earliest published computer algorithm.", "Ada Lovelace", 0),
    ("Isaac Newton established classical laws of universal gravitation.", "Isaac Newton", 0),
    ("Charles Darwin described the origin of diverse animal species.", "Charles Darwin", 0),
    ("Nikola Tesla invented the alternating current induction motor.", "Nikola Tesla", 0),
    ("Richard Feynman contributed fundamentally to quantum electrodynamics.", "Richard Feynman", 0),

    ("The international conference was hosted downtown in Paris.", "Paris", 1),
    ("The research institute is headquartered in London.", "London", 1),
    ("Many technological startups emerged across Tokyo.", "Tokyo", 1),
    ("The diplomatic summit took place peacefully in Geneva.", "Geneva", 1),
    ("The astronomical observatory operates in Hawaii.", "Hawaii", 1),
    ("The financial hub expanded significantly throughout New York.", "New York", 1),
    ("Researchers gathered at a historic university in Berlin.", "Berlin", 1),
    ("The trade delegation arrived safely in Singapore.", "Singapore", 1),

    ("Google released powerful artificial intelligence platforms.", "Google", 2),
    ("Microsoft announced strategic cloud computing investments.", "Microsoft", 2),
    ("NASA launched robotic exploration rovers toward Mars.", "NASA", 2),
    ("CERN operates the massive Large Hadron Collider accelerator.", "CERN", 2),
    ("Tesla deployed autonomous driving driver assistance software.", "Tesla", 2),
    ("Apple manufactures consumer electronic hardware and laptops.", "Apple", 2),
    ("Amazon manages distributed global logistics operations.", "Amazon", 2),
    ("IBM develops quantum computing processing hardware.", "IBM", 2),
]

# Semantic Textual Similarity (STS) pairs with human gold scores [0.0 - 5.0]
STS_PAIRS = [
    ("A man is playing a guitar on stage.", "A person is playing an acoustic instrument.", 4.5),
    ("The chef prepared a gourmet dinner.", "A cook made a delicious evening meal.", 4.6),
    ("Children are running across the grass.", "Kids are playing outdoors on the lawn.", 4.4),
    ("The airplane landed at the international airport.", "The aircraft touched down on the runway.", 4.7),
    ("A cat is sleeping on the couch.", "A feline is resting comfortably on a sofa.", 4.8),
    ("The scientist published research findings.", "An investigator shared experimental results.", 4.2),
    ("He drove his car to work in the morning.", "She commuted to her office by vehicle.", 3.5),
    ("The stock market crashed after the announcement.", "Equities declined sharply following the news.", 4.3),
    ("The weather was hot and sunny all afternoon.", "A warm sun shone brightly during the day.", 4.2),
    ("A dog barked at the stranger.", "The hound made loud noises at an unfamiliar person.", 4.0),

    # Low similarity pairs
    ("A man is playing a guitar on stage.", "The chef prepared a gourmet dinner.", 0.6),
    ("Children are running across the grass.", "The airplane landed at the airport.", 0.4),
    ("A cat is sleeping on the couch.", "The stock market crashed yesterday.", 0.2),
    ("He drove his car to work in the morning.", "Photosynthetic plants convert sunlight.", 0.1),
    ("The scientist published research findings.", "The dog barked at the stranger.", 0.8),
    ("The weather was hot and sunny all afternoon.", "Deep neural networks learn representations.", 0.3),
    ("Commercial banks reported quarterly earnings.", "Wildflowers bloomed along the river bank.", 0.9),
    ("The airplane landed on the runway.", "Children threw stones into the current.", 0.5),
]


def run_full_benchmark_suite(
    models: list[str] = [
        "bert-base-uncased",
        "roberta-base",
        "gpt2",
        "distilgpt2",
    ],
    output_dir: str = "benchmark_results",
) -> dict:
    """Run the complete empirical representation geometry benchmark across models."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    benchmark_data: dict[str, object] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_sentences": len(BENCHMARK_SENTENCES),
            "models_evaluated": models,
            "research_question": "Does representation geometry stabilization correspond to linguistic probe usability?",
        },
        "models": {},
    }

    for model_name in models:
        print(f"\n========================================================")
        print(f" Auditing Architecture: {model_name}")
        print(f"========================================================")

        # 1. Audit representation geometry
        auditor = RepresentationAuditor(model_name=model_name, batch_size=16)
        results = auditor.audit(
            sentences=BENCHMARK_SENTENCES,
            token_sentences=POLYSEMOUS_TOKEN_SENTENCES,
            labels=TOPIC_LABELS,
            keep_representations=True,
            sod_threshold=0.50,
            sod_window=2,
        )

        layer_reps = results.layer_representations
        n_layers_total = layer_reps.shape[0]

        # 2. Evaluate SOD sensitivity grid
        sod_grid = evaluate_sod_sensitivity(layer_reps)

        # 3. Systematic Diagnostic Probing Suite
        print(f"  -> Running 4-task diagnostic probing suite...")

        # Task 1: Topic Classification (Sentence-level Semantics)
        topic_probe = evaluate_linear_classification_probe(
            layer_representations=layer_reps,
            labels=TOPIC_LABELS,
            task_name="Topic Classification",
            dimension="semantic",
            metric_name="Macro F1",
        )

        # Task 2: POS Probing (Syntactic)
        pos_sentences = [item[0] for item in POS_PROBING_DATA]
        pos_words = [item[1] for item in POS_PROBING_DATA]
        pos_labels = np.array([item[2] for item in POS_PROBING_DATA])

        pos_reps_list = []
        valid_pos_labels = []
        extractor = auditor._extractor
        for text, word, lbl in zip(pos_sentences, pos_words, pos_labels):
            tok_rep = extractor.extract_token_representations([text], word)
            if tok_rep.shape[1] > 0:
                pos_reps_list.append(tok_rep[:, 0, :])  # (n_layers+1, hidden)
                valid_pos_labels.append(lbl)

        if pos_reps_list:
            pos_layer_reps = np.stack(pos_reps_list, axis=1)  # (n_layers+1, n_valid, hidden)
            pos_probe = evaluate_linear_classification_probe(
                layer_representations=pos_layer_reps,
                labels=np.array(valid_pos_labels),
                task_name="Part-of-Speech Tagging",
                dimension="syntactic",
                metric_name="Accuracy",
            )
        else:
            pos_probe = topic_probe  # Fallback

        # Task 3: NER Probing (Entity)
        ner_sentences = [item[0] for item in NER_PROBING_DATA]
        ner_entities = [item[1] for item in NER_PROBING_DATA]
        ner_labels = np.array([item[2] for item in NER_PROBING_DATA])

        ner_reps_list = []
        valid_ner_labels = []
        for text, ent, lbl in zip(ner_sentences, ner_entities, ner_labels):
            tok_rep = extractor.extract_token_representations([text], ent)
            if tok_rep.shape[1] > 0:
                ner_reps_list.append(tok_rep[:, 0, :])
                valid_ner_labels.append(lbl)

        if ner_reps_list:
            ner_layer_reps = np.stack(ner_reps_list, axis=1)
            ner_probe = evaluate_linear_classification_probe(
                layer_representations=ner_layer_reps,
                labels=np.array(valid_ner_labels),
                task_name="Named Entity Recognition",
                dimension="entity",
                metric_name="Accuracy",
            )
        else:
            ner_probe = topic_probe

        # Task 4: STS Similarity Probe (Sentence Semantics)
        sts_sent_a = [pair[0] for pair in STS_PAIRS]
        sts_sent_b = [pair[1] for pair in STS_PAIRS]
        sts_gold = np.array([pair[2] for pair in STS_PAIRS])

        reps_a = extractor.extract(sts_sent_a)
        reps_b = extractor.extract(sts_sent_b)
        sts_probe = evaluate_sts_similarity_probe(reps_a, reps_b, sts_gold)

        probing_suite = [pos_probe, ner_probe, topic_probe, sts_probe]

        # 4. Compare geometric markers vs probing plateaus
        comparison = compare_geometry_against_probing(
            sod_layer=results.semantic_onset_depth,
            cka_elbow_layer=results.cka_elbow_layer,
            max_drift_layer=results.max_drift_layer,
            probing_results=probing_suite,
        )

        print(f"\n  [Audit Summary - {model_name}]")
        print(f"  Avg Anisotropy        : {results.avg_anisotropy:.4f}")
        print(f"  SOD (Hypothesis)      : layer {results.semantic_onset_depth}")
        print(f"  CKA Elbow             : layer {results.cka_elbow_layer}")
        print(f"  Max Drift             : layer {results.max_drift_layer}")
        print(f"  Avg Probe Plateau     : layer {comparison['average_probe_plateau']:.1f}")
        print(f"  MAE (SOD vs Probes)   : {comparison['mae_sod_vs_probes']:.2f} layers")
        print(f"  MAE (CKA vs Probes)   : {comparison['mae_cka_vs_probes']:.2f} layers")
        print(f"  MAE (Drift vs Probes) : {comparison['mae_drift_vs_probes']:.2f} layers")

        model_record = {
            "model_name": model_name,
            "n_layers": results.n_layers,
            "avg_anisotropy": results.avg_anisotropy,
            "min_anisotropy_layer": results.min_anisotropy_layer,
            "max_mev_layer": results.max_mev_layer,
            "max_svd_ratio_layer": results.max_svd_ratio_layer,
            "sod_layer": results.semantic_onset_depth,
            "cka_elbow_layer": results.cka_elbow_layer,
            "max_drift_layer": results.max_drift_layer,
            "metrics": {
                "anisotropy_per_layer": results.anisotropy_per_layer,
                "uncentered_svd_ratio_per_layer": results.uncentered_svd_ratio_per_layer,
                "mev_per_layer": results.mev_per_layer,
                "layer_drift_cosine": results.layer_drift_cosine,
                "layer_drift_cka": results.layer_drift_cka,
                "self_similarity_per_layer": results.self_similarity_per_layer,
                "silhouette_per_layer": results.silhouette_per_layer,
            },
            "sod_sensitivity": sod_grid,
            "probing": {
                p.task_name: {
                    "dimension": p.dimension,
                    "metric_name": p.metric_name,
                    "scores_per_layer": p.scores_per_layer,
                    "peak_layer": p.peak_layer,
                    "peak_score": p.peak_score,
                    "plateau_layer": p.plateau_layer,
                }
                for p in probing_suite
            },
            "probing_vs_geometry_comparison": comparison,
        }

        benchmark_data["models"][model_name] = model_record

    # Save to disk
    json_path = Path(output_dir) / "benchmark_data.json"
    json_path.write_text(json.dumps(benchmark_data, indent=2))
    print(f"\n[Success] Benchmark results successfully saved to: {json_path}")
    return benchmark_data


if __name__ == "__main__":
    run_full_benchmark_suite()
