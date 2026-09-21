# Tokenize Once, Recommend Anywhere: Unified Item Tokenization for Multi-domain LLM-based Recommendation

Yu Hou \( {}^{1} \) , Won-Yong Shin \( {}^{1 * } \)

\( {}^{1} \) Yonsei University

\{houyu, wy.shin\}@yonsei.ac.kr

## Abstract

Large language model (LLM)-based recommender systems have achieved high-quality performance by bridging the discrepancy between the item space and the language space through item tokenization. However, existing item tokeniza-tion methods typically require training separate models for each item domain, limiting generalization. Moreover, the diverse distributions and semantics across item domains make it difficult to construct a unified tokenization that preserves domain-specific information. To address these challenges, we propose UniTok, a Unified item Tokenization framework that integrates our own mixture-of-experts (MoE) architecture with a series of codebooks to convert items into discrete tokens, enabling scalable tokenization while preserving semantic information across multiple item domains. Specifically, items from different domains are first projected into a unified latent space through a shared encoder. They are then routed to domain-specific experts to capture the unique semantics, while a shared expert, which is always active, encodes common knowledge transferable across domains. Additionally, to mitigate semantic imbalance across domains, we present a mutual information calibration mechanism, which guides the model towards retaining similar levels of semantic information for each domain. Comprehensive experiments on wide-ranging real-world datasets demonstrate that the proposed UniTok framework is (a) highly effective: achieving up to 51.89% improvements over strong benchmarks, (b) theoretically sound: showing the analytical validity of our architectural design and optimization; and (c) highly generalizable: demonstrating robust performance across diverse domains without requiring per-domain retraining, a capability not supported by existing baselines.

Code - https://github.com/jackfrost168/UniTok

## Introduction

Large language models (LLMs) have recently become a promising paradigm for generative recommendation (Rajput et al. 2023; Hua et al. 2023), leveraging their strong generalization, language understanding, and world knowledge to support personalized recommendation beyond traditional language processing tasks. To effectively use LLMs for recommendation, items must be indexed using identifiers, a process known as item tokenization (Rajput et al. 2023). Item tokenization converts items into discrete tokens, such as ID-based representations (Hua et al. 2023), textual descriptors (Zhang et al. 2021), or codebook-based identifiers (Rajput et al. 2023). This bridges the gap between the item space and the language space, enabling LLMs to process items as part of natural language sequences and making generative recommendations feasible.

Existing item tokenization methods (Rajput et al. 2023; Wang et al. 2024) are primarily tailored to items in single-domain settings, necessitating the training of separate to-kenizers for each item domain (hereafter, "domain" refers to "item domain" for brevity). In practice, this domain-specific design aligns with the fact that recommender systems are often deployed independently per domain; thus, users rarely perceive quality issues. However, as recommendation tasks increasingly span multiple domains, such as diverse item categories or services, this siloed approach leads to inefficiencies in training, deployment, and maintenance, ultimately hindering scalability. In contrast, other machine learning fields have seen a growing shift towards building unified models for multi-domain learning, driven by the need to reduce redundant training, improve parameter efficiency, and facilitate knowledge sharing across domains. Notable advances in this direction have been achieved in language processing (Gururangan et al. 2020; Raffel et al. 2020) and computer vision (Ullah et al. 2022; Jain et al. 2023), demonstrating the feasibility and value of such generalization.

Inspired by this, a natural question arising is: "How can we design a unified item tokenization framework for LLM-based recommendation that can be effectively generalized across multiple domains with minimal computational overhead?" To answer this question, we would like to outline the following two design challenges:

- C1. Training overhead: Repeatedly training domain-specific tokenizers is inefficient and resource-intensive. As shown in Figure 1a, when applied to 10 distinct domains, our UniTok method reduces the total number of trainable parameters by \( {9.63} \times \) compared to codebook-based item tokenization methods (Rajput et al. 2023; Wang et al. 2024), which require training a separate set of codebooks for each dataset to quantize items.

- C2. Semantic alignment: The tokenizer must capture rich semantics from diverse domains; however, naïvely using a shared token space across domains can cause semantic mixing and biased token assignments. Figure 1b exemplifies this challenge.

---

*Corresponding Author.

Copyright © 2026, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

---

![1_170_155_675_232_0.jpg](images/1_170_155_675_232_0.jpg)

Figure 1: Examples illustrating (a) a comparison of the number of trainable parameters between codebook-based item tokenization methods and our method, UniTok, when applied to 10 distinct item domains, and (b) the inherent challenge of item tokenization across multiple domains.

To address these aforementioned challenges, we make the first attempt towards developing a Unified item Tokenization framework designed to work effectively across multiple domains, named UniTok.

(Idea 1): Different domains often have exhibit distinct data distributions, requiring models to be trained separately to capture domain-specific patterns. To move beyond this limitation, we aim to design a unified item tokenization model capable of handling multiple domains without losing domain-specific knowledge. Achieving this goal requires the model to internally disentangle domain-specific learning from shared representations. To this end, we propose a new mixture-of-experts (MoE) architecture, dubbed Token-MoE, wherein domain-specific experts specialize in modeling patterns unique to each domain, while a shared expert captures the common knowledge across multiple domains. This architectural design enables the unified model to retain domain specialization without sacrificing global knowledge sharing (solving C1 and partially contributing to C2).

(Idea 2): To preserve item semantics during tokenization, we adopt a codebook-based approach (Rajput et al. 2023) embedded within our MoE architecture, allowing each expert to specialize in distinct semantic patterns. However, the central challenge in multi-domain settings lies in ensuring semantic balance across diverse domains (Ma et al. 2022). To address this, we introduce a mutual information (MI) calibration mechanism that explicitly encourages latent embed-dings from each domain to retain sufficient and consistent informativeness. By minimizing the variance of MI across item domains, this mechanism attenuates inter-domain performance variability, enabling more stable and consistent generalization across diverse semantic spaces (solving C2).

Our main contributions are summarized as follows:

- New methodology: We propose UniTok, a unified item tokenization framework that integrates our customized MoE with codebooks to extract the semantic tokens for items across multiple domains while maintaining semantic balance.

- Extensive evaluations: Through comprehensive experimental evaluations on diverse real-world datasets, we demonstrate (a) the superiority of UniTok in multi-domain scenarios, achieving substantial improvements of up to 51.89% in NDCG@10, (b) the efficiency of Uni-Tok, with a \( {9.63} \times \) reduction in model size compared to competitors, and (c) the strong generalization capability of UniTok, exhibiting robust performance in zero-shot settings without additional retraining.

- Theoretical justifications: We theoretically prove that UniTok (a) induces a higher entropy token space, (b) achieves a lower quantization error, and (c) ensures semantic consistency across domains by reducing the MI variance, fostering stable and balanced performance.

We refer readers to the supplementary materials for technical details omitted due to page limits.

## Preliminaries

## Mixture-of-Experts

The mixture-of-experts (MoE) architecture (Jacobs et al. 1991; Shazeer et al. 2017; Fedus, Zoph, and Shazeer 2022; Dai et al. 2024) consists of multiple expert networks, each specializing in handling different parts of the input space. Formally, given an input \( \mathbf{x} \) , the output of an MoE model is a weighted combination of expert modules: \( \operatorname{MoE}\left( \mathbf{x}\right)  = \; \mathop{\sum }\limits_{{k = 1}}^{K}{G}_{k}\left( \mathbf{x}\right) {E}_{k}\left( \mathbf{x}\right) \) , where \( K \) is the number of experts, \( {E}_{k}\left( \mathbf{x}\right) \) denotes the output of the \( k \) -th expert, and \( {G}_{k}\left( \mathbf{x}\right) \) is a softmax-based router function that assigns a probability to the \( k \) -th expert for given input. By dynamically routing input to specialized experts, MoE models can achieve greater model capacity and computational efficiency.

The MoE architecture is not only effective for model scaling but also well-suited for training on a large mixture of datasets (Jain et al. 2023). Its gating mechanism enables adaptive expert selection for each dataset, allowing the model to generalize across diverse sources while minimizing interference between distributions.

## Codebook-based Identifiers

Codebook-based identifiers (Rajput et al. 2023) are generated using residual quantization (RQ), which encodes item metadata into hierarchical code sequences by sequentially applying codebooks to residuals across multiple stages. Given an input vector \( \mathbf{x} \) , the process starts with an initial residual \( {\mathbf{r}}^{\left( 0\right) } = \mathbf{x} \) , and RQ recursively quantizes it using a series of \( L \) codebooks \( \left\{  {{C}_{1},{C}_{2},\ldots ,{C}_{L}}\right\} \) , where \( {C}_{\ell } \triangleq \; {\left\{  {\mathbf{c}}_{t}\right\}  }_{t = 1}^{T} \) contains \( T \) code vectors at level \( \ell \) . The process is defined as

\[
{\mathbf{c}}_{\ell } = \arg \mathop{\min }\limits_{{\mathbf{c} \in  {C}_{\ell }}}{\begin{Vmatrix}{\mathbf{r}}^{\left( \ell  - 1\right) } - \mathbf{c}\end{Vmatrix}}^{2}, \tag{1}
\]

\[
{\mathbf{r}}^{\left( \ell \right) } = {\mathbf{r}}^{\left( \ell  - 1\right) } - {\mathbf{c}}_{\ell }, \tag{2}
\]

where \( {\mathbf{r}}^{\left( \ell \right) } \) is the residual at level \( l \) , and \( {\mathbf{c}}_{\ell } \) is the nearest code vector selected from codebook \( {C}_{\ell } \) . The final approximation of \( \mathbf{x} \) after \( L \) quantization stages is given by \( \widehat{\mathbf{x}} = \mathop{\sum }\limits_{{\ell  = 1}}^{L}{\mathbf{c}}_{\ell } \) .

Each selected code vector \( {\mathbf{c}}_{\ell } \) corresponds to a discrete index \( {z}_{\ell } \in  \{ 1,\cdots , T\} \) , resulting in a token sequence \( \left( {{z}_{1},\cdots ,{z}_{L}}\right) \) that compactly represents the original input. This hierarchical quantization mechanism provides the foundation for our codebook-based identifiers, offering compact and semantically meaningful tokens well-suited for item tokenization. In recommender systems, such discrete tokens can then be directly used as input to LLMs (Rajput et al. 2023; Wang et al. 2024), bridging the gap between item and language spaces while preserving semantic structure.

![2_187_150_1428_595_0.jpg](images/2_187_150_1428_595_0.jpg)

Figure 2: The schematic overview of the proposed UniTok framework.

## Methodology

## Task Formulation

Multi-domain setting. Let \( \mathcal{D} = \left\{  {{\mathcal{D}}_{1},\ldots ,{\mathcal{D}}_{K}}\right\} \) be a mixture of recommendation datasets for \( K \) distinct item domains, where each domain \( {\mathcal{D}}_{k} \) contains an item set \( {\mathcal{I}}_{k} \) with associated textual metadata, such as titles, categories, features. Unlike typical single-domain scenarios, multi-domain settings pose significant challenges due to distributional inconsistencies (Jain et al. 2023), making it crucial to handle such variability in item tokenization. Addressing this issue is essential for developing a scalable and unified item tokeniza-tion method that supports multiple domains, particularly in LLM-based recommender systems. Although collaborative signals can enhance recommendation performance, they inherently rely on user-item interactions, introducing computational overhead and limiting generalization when shared users are absent across domains. Instead, we aim to develop a unified tokenization method that operates independently of user data, enabling scalable and general-purpose recommendations in multi-domain LLM-based systems.

Item tokenization task. Given raw items \( {\mathcal{I}}_{k} \) from any domain-specific dataset \( {\mathcal{D}}_{k} \) , we assume that we have access to a pre-trained content encoder to generate the semantic em-beddings for domain \( k \) , denoted as \( {\mathbf{X}}^{k} \in  {\mathbb{R}}^{\left| {\mathcal{I}}_{k}\right|  \times  d} \) , where \( \left| {\mathcal{I}}_{k}\right| \) is the number of items in domain \( k \) and \( d \) is the embedding dimensionality. The \( i \) -th embedded item is represented as \( {\mathbf{x}}_{i}^{k} \in  {\mathbf{X}}^{k} \) . The objective is to learn a mapping function \( \mathcal{F} : {\mathbb{R}}^{d} \rightarrow  \mathcal{C} \) that projects each continuous embedding \( {\mathbf{x}}_{i}^{k} \) into a discrete codeword \( {\mathbf{c}}_{i}^{k} \in  \mathcal{C} \) , where \( \mathcal{C} \) denotes the shared space of discrete item tokens across all domains.

## Overview of UniTok

We recall that recent item tokenization methods for LLM-based recommendations focus merely on a single-domain setting (Rajput et al. 2023; Wang et al. 2024; Zheng et al. 2024). Howerer, real-world systems recently operate across multiple domains (Jiang et al. 2022; Ning et al. 2023), leading to repeated training and semantic inconsistency across domains-an issue largely overlooked by prior studies.

To tackle challenges C1 and C2 in Section 1, UniTok leverages a shared autoencoder to project items in the mixture of domains into a unified latent space. To achieve effective tokenization across diverse domains, we present Token-MoE with codebook-based identifiers, an MoE architecture composed of domain-specific experts that capture specialized semantics and a shared expert that encodes generalized knowledge. Additionally, we present an MI-based loss that enforces consistent semantics across multiple domains by regulating the informativeness of latent embeddings.

## Architectural Details

As illustrated in Figure 2, our UniTok consists of four key components: a shared autoencoder, TokenMoE, codebook-based identifiers, and an MI calibration mechanism.

Shared autoencoder. Given semantic embeddings from the mixture of domains, we first employ a shared autoen-coder composed of an encoder \( {f}_{\theta } \) , to project items into a unified latent space, and decoder \( {g}_{\phi } \) to reconstruct the semantic embeddings (see the coral pink region in Figure 2). This establishes a unified representation space across multiple domains, which captures common structural patterns while retaining essential information for reconstruction.

Formally, for each input item \( {\mathbf{x}}_{i}^{k} \in  {\mathbf{X}}^{k} \) from domain \( {\mathcal{D}}_{k} \) , the encoder produces a latent embedding \( {\mathbf{z}}_{i}^{k} = {f}_{\theta }\left( {\mathbf{x}}_{i}^{k}\right) \) , and the decoder reconstructs the input item as \( {\widehat{\mathbf{x}}}_{i}^{k} = {g}_{\phi }\left( {\widehat{\mathbf{z}}}_{i}^{k}\right) \) , where \( {\widehat{\mathbf{z}}}_{i}^{k} = \operatorname{TokenMoE}\left( {\mathbf{z}}_{i}^{k}\right) \) (to be specified in Eq. (5)). The model is optimized using the reconstruction loss:

\[
{\mathcal{L}}_{\text{ Rec }} = \mathop{\sum }\limits_{{k = 1}}^{K}\mathop{\sum }\limits_{{{\mathbf{x}}_{i}^{k} \in  {\mathbf{X}}^{k}}}{\begin{Vmatrix}{\mathbf{x}}_{i}^{k} - {\widehat{\mathbf{x}}}_{i}^{k}\end{Vmatrix}}^{2}, \tag{3}
\]

where \( \parallel  \cdot  \parallel \) denotes the \( {L}_{2} \) norm. This reconstruction loss shapes the latent space into a compact and informative representation that retains core semantics for item tokenization.

TokenMoE. Conventional tokenization models applied to a unified item space often fail to capture domain-specific nuances, as they treat diverse domains uniformly, potentially leading to the loss of specialized semantic information. To address this limitation, we present TokenMoE, a more generalizable MoE architecture, which routes items to both domain-specific experts and a shared expert. This design enables domain-specific experts to learn specialized patterns, while a shared expert with always-active routing facilitates efficient knowledge transfer across multiple domains. Distinct from earlier approaches that utilize MoE within the feedforward layers of transformers (Dai et al. 2024), our key contribution lies in uniquely integrating MoE into the tok-enization module to better enable domain-aware tokeniza-tion. The TokenMoE module is illustrated in the light blue region of Figure 2 (with one of the domain-specific expert highlighted in purple and the shared expert in orange). This design addresses C1 and partially mitigates C2.

Specifically, after encoding, the item latent embedding \( {\mathbf{z}}_{i}^{k} \) passes through a router function \( G\left( \cdot \right) \) , which produces a softmax distribution over \( K \) domain-specific experts: \( G\left( {\mathbf{z}}_{i}^{k}\right)  = \left\{  {{G}_{1},{G}_{2},\ldots ,{G}_{K}}\right\} \) , where each \( {G}_{k} \) is computed as

\[
{G}_{k} = \frac{\exp \left( {s}_{i}^{\left( k\right) }\right) }{\mathop{\sum }\limits_{{j = 1}}^{K}\exp \left( {s}_{i}^{\left( j\right) }\right) },{s}_{i} = h\left( {\mathbf{z}}_{i}^{k}\right)  \in  {\mathbb{R}}^{K}, \tag{4}
\]

and \( h\left( \cdot \right) \) is a learnable linear transformation in the router producing the router logits \( {s}_{i} \) , and \( {s}_{i}^{\left( k\right) } \) denotes the logit corresponding to the \( k \) -th domain-specific expert. Here, \( K \) is the total number of domain-specific experts, which is typically aligned with the number of domains.

The item is then routed to the top- \( N \) domain-specific experts \( {}^{1} \) based on the highest values of \( G\left( {\mathbf{z}}_{i}^{k}\right) \) , while it is also deterministically assigned to a shared expert. Each expert, including the shared one, is implemented as a codebook-based identifier (to be specified later). The final \( {\widehat{\mathbf{z}}}_{i}^{k} \) is computed as a weighted combination of the selected domain-specific experts and the shared expert, which is formulated as follows:

\[
\left\{  \begin{array}{l} {\widehat{\mathbf{z}}}_{i}^{k} = \operatorname{TokenMoE}\left( {\mathbf{z}}_{i}^{k}\right)  = \mathop{\sum }\limits_{{k = 1}}^{K}{G}_{k}{E}_{k}\left( {\mathbf{z}}_{i}^{k}\right)  + {E}_{\text{ share }}\left( {\mathbf{z}}_{i}^{k}\right) , \\  {G}_{k} = \left\{  \begin{array}{ll} {G}_{k}, & \text{ if }k \in  {\operatorname{Top}}_{N}\left( {G\left( {\mathbf{z}}_{i}^{k}\right) }\right) \\  0, & \text{ otherwise } \end{array}\right.  \end{array}\right.
\]

(5)where, \( {E}_{k}\left( \cdot \right) \) denotes the \( k \) -th expert module, \( {E}_{\text{ share }}\left( \cdot \right) \) denotes the shared expert module, and \( {\operatorname{Top}}_{N}\left( {G\left( {\mathbf{z}}_{i}^{k}\right) }\right)  = \; \left\{  {{k}_{1},{k}_{2},\ldots ,{k}_{N}}\right\} \) is the set of indices corresponding to the top- \( N \) experts selected by the router. Figure 2 shows an example where the top-1 expert is selected.

TokenMoE routes each item to domain-specific experts via its learned router, while a shared expert captures common knowledge through a deterministic path. To encourage expert specialization, each expert is initialized with the mean feature of a specific domain (Wang et al. 2024), providing a strong inductive bias for domain-aware tokenization without requiring explicit supervision. By activating only a subset of experts per item, TokenMoE enhances scalability, maintains domain specificity, and supports better generalization.

Codebook-based identifiers. We adopt RQ (Rajput et al. 2023) in each expert to discretize each item into compact token sequences. Given an item latent embedding \( {\mathbf{z}}_{i}^{k} \in  {\mathbb{R}}^{d} \) produced by the shared encoder, RQ approximates it through a sequence of codebooks \( \left\{  {{C}_{1},{C}_{2},\ldots ,{C}_{L}}\right\} \) , where \( L \) is the codebook size, and each codebook \( {C}_{\ell } \triangleq  {\left\{  {\mathbf{c}}_{t}\right\}  }_{t = 1}^{T} \) contains \( T \) code vectors. As shown in the bottom-left of Figure 2, at each level \( \ell \) , the residual \( {\mathbf{r}}^{\left( \ell \right) } \) from the previous step is encoded using the nearest code \( {\mathbf{c}}_{\ell } \) in \( {C}_{\ell } \) . The sum of all selected codes reconstructs the original latent embedding:

\[
{E}_{k}\left( {\mathbf{z}}_{i}^{k}\right)  \approx  \mathop{\sum }\limits_{{\ell  = 1}}^{L}{\mathbf{c}}_{\ell },\;\text{ where }{\mathbf{c}}_{\ell } \in  {C}_{\ell }. \tag{6}
\]

This hierarchical quantization enables fine-grained and memory-efficient tokenization, as each item is tokenized by a discrete codeword:

\[
{\mathbf{z}}_{i}^{k} \mapsto  {\mathbf{c}}_{i}^{k} = \left( {{z}_{1},\ldots ,{z}_{L},{e}_{1},\ldots ,{e}_{N}}\right) , \tag{7}
\]

where \( {z}_{\ell } \in  \{ 1,\ldots , T\} \) denotes the index of selected code vector \( {\mathbf{c}}_{\ell } \) from the \( \ell \) -th codebook \( {C}_{\ell } \) and \( {e}_{n} \in  \{ 1,\ldots , K\} \) indicates the expert ID of the \( n \) -th top- \( N \) expert chosen by the router.

To train this quantization process, we adopt the RQ loss:

\[
{\mathcal{L}}_{\mathrm{{RQ}}} \mathrel{\text{ := }} \mathop{\sum }\limits_{{\ell  = 1}}^{L}{\begin{Vmatrix}\operatorname{sg}\left\lbrack  {\mathbf{r}}^{\left( \ell \right) }\right\rbrack   - {\mathbf{c}}_{\ell }\end{Vmatrix}}^{2} + \alpha {\begin{Vmatrix}{\mathbf{r}}^{\left( \ell \right) } - \operatorname{sg}\left\lbrack  {\mathbf{c}}_{\ell }\right\rbrack  \end{Vmatrix}}^{2}, \tag{8}
\]

where \( {\mathbf{r}}^{\left( \ell \right) } \) is the residual vector at level \( \ell ,{\mathbf{c}}_{\ell } \) is the selected code vector from \( {C}_{\ell },\operatorname{sg}\left\lbrack  \cdot \right\rbrack \) is the stop-gradient operator, and \( \alpha \) is a balancing hyperparameter. In Eq. (8), the first term aligns the code vector to the target residual (codebook learning), and the second term forces the encoder and router to commit to the selected quantized code vector.

MI calibration. As the shared encoder will project all items into a unified latent embedding space, it is not straightforward for the model to precisely capture domain-specific features due to semantic imbalance. This occurs when the quality of learned latent embeddings varies significantly across diverse domains-particularly between simple and complex domains-causing semantically similar items to be assigned inconsistent tokens (Ma et al. 2022).

As another key contribution aimed at mitigating this issue, we introduce an MI mechanism to ensure that the latent space retains sufficient information from each domain. Specifically, we adopt the Hilbert-Schmidt independence criterion (HSIC) (Gretton et al. 2005; Li et al. 2021) \( {}^{2} \) , serving as a proxy for the MI, with a higher HSIC value indicates stronger dependence. As illustrated in Figure 3, for domain \( k \) , HSIC measures the dependence between the input semantic embeddings \( {\mathbf{X}}^{k} = \left\{  {{\mathbf{x}}_{1}^{k},\ldots ,{\mathbf{x}}_{\left| {\mathcal{I}}_{k}\right| }^{k}}\right\} \) and their latent embeddings \( {\mathbf{Z}}^{k} = \left\{  {{\mathbf{z}}_{1}^{k},\ldots ,{\mathbf{z}}_{\left| {\mathcal{I}}_{k}\right| }^{k}}\right\} \) in a reproducing kernel Hilbert space (RKHS), computed as:

\[
\widehat{\operatorname{HSIC}}\left( {{\mathbf{X}}^{k},{\mathbf{Z}}^{k}}\right)  = \frac{1}{{\left( \left| {\mathcal{I}}_{k}\right|  - 1\right) }^{2}}\operatorname{Tr}\left( \mathbf{{UHVH}}\right) , \tag{9}
\]

---

\( {}^{1}N \) is typically set as either 1 or 2 to promote sparsity in expert activation, which significantly reduces computational overhead while preserving model capacity (Lepikhin et al. 2021; Fedus, Zoph, and Shazeer 2022).

---

![4_157_152_707_245_0.jpg](images/4_157_152_707_245_0.jpg)

Figure 3: The illustration of MI calibration.

where \( \mathbf{U},\mathbf{V} \in  {\mathbb{R}}^{\left| {\mathcal{I}}_{k}\right|  \times  \left| {\mathcal{I}}_{k}\right| } \) are Gaussian kernel matrices computed over \( {\mathbf{X}}^{k} \) and \( {\mathbf{Z}}^{\mathbf{k}} \) , respectively \( {}^{3} \) . The centering matrix \( \mathbf{H} = \mathbf{I} - \frac{1}{\left| {\mathcal{I}}_{k}\right| }{\mathbf{{11}}}^{\top } \) ensures zero-mean embeddings in RKHS; and \( \operatorname{Tr}\left( \mathbf{{UHVH}}\right) \) computes the Hilbert-Schmidt norm of the cross-covariance between the two RKHSs. This design addresses C2.

To enforce semantic balance across multiple domains (see Figure 3), we characterize the MI calibration loss as:

\[
{\mathcal{L}}_{\mathrm{{MI}}} = \operatorname{Var}\left\lbrack  {\widehat{I}}^{\left( k\right) }\right\rbrack   - \beta \mathbb{E}\left\lbrack  {\widehat{I}}^{\left( k\right) }\right\rbrack  , \tag{10}
\]

where \( {\widehat{I}}^{\left( k\right) } = \widehat{\operatorname{HSIC}}\left( {{\mathbf{X}}^{k},{\mathbf{Z}}^{k}}\right) \) and \( \beta \) is a weighting hyperpa-rameter. The first term penalizes high variance of MI across domains to mitigate semantic imbalance, while the second term enforces each domain to retain sufficient domain-specific information.

Optimization. We train the model using the overall loss:

\[
{\mathcal{L}}_{\text{ total }} = {\mathcal{L}}_{\text{ Rec }} + {\lambda }_{\mathrm{{RQ}}}{\mathcal{L}}_{\mathrm{{RQ}}} + {\lambda }_{\mathrm{{MI}}}{\mathcal{L}}_{\mathrm{{MI}}}, \tag{11}
\]

where \( {\lambda }_{\mathrm{{RQ}}} \) and \( {\lambda }_{\mathrm{{MI}}} \) are hyper-parameters to control the strength of RQ and MI, respectively.

To instantiate UniTok in LLM-based recommender systems, we first train UniTok on all items using Eq. (11). The trained UniTok then tokenizes each item into discrete semantic tokens, enabling LLMs to operate in the token space. Following prior work (Wang et al. 2024), user interaction histories \( \mathbf{u} \) are converted into item token sequences \( \mathbf{u} = \left\lbrack  {{\widetilde{\mathbf{c}}}_{1},{\widetilde{\mathbf{c}}}_{2},\ldots ,{\widetilde{\mathbf{c}}}_{P}}\right\rbrack \) , and the LLM-based recommender system learns to predict the next interacted item token \( {\widetilde{\mathbf{c}}}_{P + 1} \) .

## Theoretical Analyses

In this subsection, we establish the following theorems, which analyze and support the effectiveness of the key components within UniTok.

Theorem 1. The token space induced by UniTok exhibits strictly higher entropy than that of standard codebook-based methods:

\[
H\left( {\mathcal{C}}_{\text{ UniTok }}\right)  > H\left( {\mathcal{C}}_{\text{ standard }}\right) , \tag{12}
\]

where \( {\mathcal{C}}_{\text{ UniTok }} \) and \( {\mathcal{C}}_{\text{ standard }} \) denote the discrete token distributions generated by UniTok and standard codebook-based methods, respectively.

This indicates that judiciously incorporating multiple experts increases the overall entropy, thereby expanding the token space capacity.

Theorem 2. Let \( \mathbb{E}\left\lbrack  {\mathcal{L}}_{\text{ UniTok }}\right\rbrack \) and \( \mathbb{E}\left\lbrack  {\mathcal{L}}_{\text{ standard }}\right\rbrack \) denote the expected quantization error of UniTok and standard codebook-based methods using a single shared codebook, respectively. Then, the following inequality holds:

\[
\mathbb{E}\left\lbrack  {\mathcal{L}}_{\text{ UniTok }}\right\rbrack   \leq  \mathbb{E}\left\lbrack  {\mathcal{L}}_{\text{ standard }}\right\rbrack \tag{13}
\]

This implies that a lower expected quantization error reflects more precise modeling of item tokenization. The To-kenMoE framework further reduces this error by leveraging expert specialization, which effectively compensates for domain-specific inaccuracies to some extent, thereby improving quantization quality across diverse domains.

Theorem 3. Suppose that the loss \( {\mathcal{L}}^{\left( k\right) } \) on the \( k \) -th domain is Lipschitz-continuous with respect to the informativeness of representations. Then, the performance variability across domains is upper-bounded by the variance of MI:

\[
\left| {{\mathcal{L}}^{\left( i\right) } - {\mathcal{L}}^{\left( j\right) }}\right|  \leq  C\sqrt{\operatorname{Var}\left\lbrack  {\widehat{I}}^{\left( k\right) }\right\rbrack  },\forall i, j \tag{14}
\]

where \( \operatorname{Var}\left\lbrack  {\widehat{I}}^{\left( k\right) }\right\rbrack \) is the variance of MI estimates across domains and \( C \) is a constant.

This implies that reducing MI variance across domains promotes consistent semantic representations, leading to more stable and reliable downstream performance.

We empirically validate each theorem's technical correctness and practical relevance through extensive experiments.

## Experimental Evaluation

In this section, we carry out comprehensive experiments to empirically validate the effectiveness of UniTok across multiple domains.

## Experimental Settings

Datasets. We conduct our experiments on ten real-world datasets spanning ten domains widely adopted for evaluating the performance of recommendations, which include Beauty, Cellphones, Grocery, Instruments, Office, Pet Supplies, Tools, Toys, Games \( {}^{4} \) , and Yelp \( {}^{5} \) . Each item contains metadata, including title, category, and features. Due to page limitations, we report complete results across all datasets to assess overall performance, while employing three representative datasets for the efficiency and ablation analyses.

---

\( {}^{2} \) Other methods, such as MINE (Belghazi et al. 2018) and In-foNCE (Oord, Li, and Vinyals 2018), can also estimate MI using neural networks, but require additional training. To maintain simplicity, we adopt HSIC, which is non-parametric.

\( {}^{3} \) The kernel matrices are computed element-wise using the Gaussian kernel: \( {U}_{ij} = u\left( {{\mathbf{x}}_{i}^{k},{\mathbf{x}}_{j}^{k}}\right)  = \exp \left( {-{\begin{Vmatrix}{\mathbf{x}}_{i}^{k} - {\mathbf{x}}_{j}^{k}\end{Vmatrix}}^{2}/2{\sigma }^{2}}\right) \) and \( {V}_{ij} = v\left( {{\mathbf{z}}_{i}^{k},{\mathbf{z}}_{j}^{k}}\right)  = \exp \left( {-{\begin{Vmatrix}{\mathbf{z}}_{i}^{k} - {\mathbf{z}}_{j}^{k}\end{Vmatrix}}^{2}/2{\sigma }^{2}}\right) \) for the bandwidth \( \sigma \) .

\( {}^{4} \) https://nijianmo.github.io/amazon/index.html.

\( {}^{5} \) https://www.yelp.com/dataset.

---

<table><tr><td>Method</td><td>Beauty</td><td>Cellphones</td><td>Grocery</td><td>Instruments</td><td>Office</td><td>Pet Supplies</td><td>Tools</td><td>Toys</td><td>Games</td><td>Yelp</td></tr><tr><td>MF</td><td>0.0369</td><td>0.0267</td><td>0.0216</td><td>0.0710</td><td>0.0255</td><td>0.0268</td><td>0.0169</td><td>0.0192</td><td>0.0366</td><td>0.0144</td></tr><tr><td>LightGCN</td><td>0.0285</td><td>0.0456</td><td>0.0357</td><td>0.0781</td><td>0.0301</td><td>0.0289</td><td>0.0257</td><td>0.0287</td><td>0.0417</td><td>0.0195</td></tr><tr><td>SASRec</td><td>0.0314</td><td>0.0446</td><td>0.0376</td><td>0.0609</td><td>0.0285</td><td>0.0301</td><td>0.0234</td><td>0.0239</td><td>0.0412</td><td>0.0183</td></tr><tr><td>Bert4Rec</td><td>0.0194</td><td>0.0268</td><td>0.0237</td><td>0.0573</td><td>0.0274</td><td>0.0161</td><td>0.0092</td><td>0.0177</td><td>0.0379</td><td>0.0131</td></tr><tr><td>P5-TID</td><td>0.0255</td><td>0.0357</td><td>0.0316</td><td>0.0721</td><td>0.0239</td><td>0.0243</td><td>0.0198</td><td>0.0202</td><td>0.0388</td><td>0.0154</td></tr><tr><td>P5-SemID</td><td>0.0304</td><td>0.0406</td><td>0.0351</td><td>0.0730</td><td>0.0283</td><td>0.0282</td><td>0.0237</td><td>0.0231</td><td>0.0432</td><td>0.0188</td></tr><tr><td>TIGER</td><td>0.0324</td><td>0.0446</td><td>0.0375</td><td>0.0788</td><td>0.0295</td><td>0.0279</td><td>0.0284</td><td>0.0268</td><td>0.0427</td><td>0.0208</td></tr><tr><td>LC-Rec</td><td>0.0381</td><td>0.0458</td><td>0.0369</td><td>0.0802</td><td>0.0311</td><td>0.0335</td><td>0.0307</td><td>0.0279</td><td>0.0451</td><td>0.0215</td></tr><tr><td>LETTER</td><td>0.0364</td><td>0.0473</td><td>0.0392</td><td>0.0831</td><td>0.0326</td><td>0.0307</td><td>0.0298</td><td>0.0291</td><td>0.0469</td><td>0.0231</td></tr><tr><td>UniTok</td><td>0.0478</td><td>0.0647</td><td>0.0533</td><td>0.0884</td><td>0.0432</td><td>0.0496</td><td>0.0439</td><td>0.0442</td><td>0.0476</td><td>0.0321</td></tr><tr><td>Gain</td><td>25.46%</td><td>36.78%</td><td>35.97%</td><td>6.38%</td><td>32.52%</td><td>48.06%</td><td>42.99%</td><td>51.89%</td><td>1.49%</td><td>38.96%</td></tr></table>

Table 1: Performance comparison among UniTok and recommendation competitors for the ten benchmark datasets in terms of NDCG@10.Here, the best and second-best performers are highlighted by bold and underline, respectively. The improvements are statistically significant on average \( \left( {p = {0.0219} < {0.05}}\right) \) based on paired t-tests over five runs across all datasets.

Competitors. To comprehensively demonstrate the superiority of UniTok, we present nine benchmark recommendation methods, including four widely-used collaborative filtering methods, MF (Rendle et al. 2009), SASRec (Kang and McAuley 2018), LightGCN (He et al. 2020), Bert4Rec (Sun et al. 2019), and five item tokenization-aided recommendations, P5-TID (Hua et al. 2023), P5-SemID (Hua et al. 2023), TIGER (Rajput et al. 2023), LC-Rec (Zheng et al. 2024), and LETTER (Wang et al. 2024).

Performance metrics. Following the full-ranking protocol (He et al. 2020), we rank all non-interacted items for each user and evaluate using Recall@ \( M\left( {\mathrm{R}@M}\right) \) and NDCG@ \( M\left( {\mathrm{\;N}@M}\right) \) , where \( M \in  \{ 5,{10}\} \) .

Implementation details. For item tokenization, we use 4-level codebook-based identifiers, where each codebook comprises 256 code vectors with a dimension of 32 . We set \( {\lambda }_{\mathrm{{RO}}} \) to 1 and \( {\lambda }_{\mathrm{{MI}}} \) to 0.03 . All experiments are conducted on two NVIDIA RTX 3090 GPUs.

## Can One Tokenizer Serve All Domains?

To evaluate the recommendation accuracy of UniTok, we compare it with benchmark item tokenization methods across ten benchmark datasets from diverse domains. Notably, distinct from benchmark methods that train separate tokenizers for each dataset, UniTok trains a single unified model that handles all ten datasets jointly. As shown in Table 1, UniTok consistently outperforms competitors across all datasets, validating the effectiveness of our unified to-kenization framework, achieving up to 51.89% improvement in terms of NDCG@10 on the Tools dataset. Unlike other approaches that require domain-specific customization, UniTok effectively captures item semantics across diverse domains through a single, shared tokenization process-highlighting the strength and generality of our proposed design.

We observe that item tokenization-aided LLM-based recommender systems, including TIGER, LC-Rec, LETTER, and UniTok, are superior to standard collaborative filtering methods such as MF, LightGCN, SASRec, and Bert4Rec. This improvement benefits from the rich semantic understanding and reasoning capabilities inherent in LLMs, which go beyond conventional user-item interactions. Moreover, integrating carefully designed item tokenization into LLM-based recommender systems yields additional performance gains compared to LLMs that solely on item metadata for tokenization (e.g., P5-TID and P5-SemID). This is because item tokenization serves as a critical bridge between the item space and the language space, allowing the underlying model to represent items in a discrete, language-aligned semantic space that enhances generalization and reasoning.

<table><tr><td>Module</td><td>Codebook-based methods</td><td>UniTok</td></tr><tr><td>Codebook</td><td>0.33M</td><td>0.36M</td></tr><tr><td>Autoencoder</td><td>87.45M</td><td>8.75M</td></tr><tr><td>Router</td><td>-</td><td>0.01M</td></tr><tr><td>Total</td><td>87.78M</td><td>9.11M</td></tr></table>

Table 2: Comparison of the number of trainable parameters. For competitors, we report the total number of trainable parameters accumulated across all ten datasets.

## Is UniTok More Efficient than Traditional Tokenizers?

To validate the efficiency of UniTok, we compare the total number of trainable parameters between UniTok and traditional codebook-based competitors, including TIGER, LC-Rec, and LETTER, which share the same underlying architecture. While these competitors require training and storing separate tokenization models for each dataset, UniTok employs a single unified model shared across all datasets. For a fair comparison, we report the cumulative trainable parameter count of the codebook-based competitors over all datasets. As shown in Table 2, UniTok achieves approximately a tenfold reduction in the total number of trainable parameters. This efficiency comes primarily from using the shared autoencoder, while the number of additional trainable parameters introduced by the codebook and Token-MoE router remains negligible. By eliminating the need for domain-specific tokenization learning, UniTok offers substantial advantages in scalability and deployment efficiency.

<table><tr><td></td><td colspan="2">Beauty</td><td colspan="2">Cellphones</td><td colspan="2">Grocery</td></tr><tr><td>Method</td><td>R@10</td><td>N@10</td><td>R@10</td><td>N@10</td><td>R@10</td><td>N@10</td></tr><tr><td>TIGER</td><td>0.0499</td><td>0.0267</td><td>0.0661</td><td>0.0342</td><td>0.0576</td><td>0.0273</td></tr><tr><td>LC-Rec</td><td>0.0564</td><td>0.0302</td><td>0.0647</td><td>0.0337</td><td>0.0584</td><td>0.0287</td></tr><tr><td>LETTER</td><td>0.0528</td><td>0.0288</td><td>0.0678</td><td>0.0363</td><td>0.0618</td><td>0.0315</td></tr><tr><td>UniTok</td><td>0.0934</td><td>0.0478</td><td>0.1251</td><td>0.0647</td><td>0.1061</td><td>0.0533</td></tr><tr><td>Gain</td><td>65.60%</td><td>58.28%</td><td>84.51%</td><td>78.23%</td><td>71.68%</td><td>69.21%</td></tr></table>

Table 3: Performance comparison of UniTok and tokeniza-tion competitors under a single unified training setup. Here, the best and second-best performers are highlighted by bold and underline, respectively.

We further evaluate the performance of competitors under a single unified training setup, where each competitor is trained jointly across ten datasets using a comparable number of trainable parameters to that of UniTok. As shown in Table 3, the competing methods exhibit substantial performance degradation in this setting compared to their single-domain scenario (see Table 1), primarily due to the difficulty of distinguishing items from different domains when using shared tokenization. In contrast, Uni-Tok maintains consistently superior recommendation performance, achieving up to 84.51% improvement in Recall@10 on Cellphones, while using a similar trainable parameter budget. This efficiency stems from its modular TokenMoE architecture, where domain-specific experts learn semantics independently, while sharing a unified token space.

## Can UniTok be Generalized to Unseen Domains?

To evaluate the generalization ability of UniTok, we adopt a zero-shot setting where our item tokenizer model, UniTok, is trained once on the ten source datasets and is directly tested on unseen datasets from three target domains-Clothing, Health, and Sports-without any additional training or fine-tuning. This setup reflects real-world scenarios where new domains may appear dynamically after the model has been deployed.

As shown in Table 4, UniTok significantly outperforms existing item tokenization-based recommender systems, achieving up to 17.87% improvement in NDCG@10 on Health. While the competitors require retraining on each new dataset to achieve reasonable tokenization results, Uni-Tok maintains robust accuracy without any further adaptation. This demonstrates our model's ability to learn a discrete token space that captures transferable item semantics across diverse domains. The consistently strong zero-shot performance further highlights the robustness and practical utility of our unified tokenization framework in supporting effective generalization across heterogeneous domains.

## What Makes UniTok Effective?

To assess the contribution of each component in UniTok, we perform an ablation study by progressively removing or modifying its core modules: the TokenMoE module, the shared expert, and the MI calibration part. UniTok-1 removes the TokenMoE and MI calibration, using only a single set of codebooks without any MoE structure; UniTok- 2 keeps TokenMoE, but removes the shared expert and MI calibration; and UniTok-3 only removes the MI calibration. UniTok includes all components. As shown in Table 8, removing any module leads to a noticeable drop in recommendation accuracy, which confirms that the combination of modules is crucial to UniTok's effectiveness. In particular, comparing UniTok-1 and UniTok-2 highlights the importance of the TokenMoE module, which significantly improves performance across all datasets by capturing domain-specific semantics. Additionally, in comparison with UniTok-3, introducing MI calibration further enhances performance by enforcing the learned latent embeddings to retain essential semantic information from each domain.

<table><tr><td></td><td colspan="2">Clothing</td><td colspan="2">Health</td><td colspan="2">Sports</td></tr><tr><td>Method</td><td>R@10</td><td>N@10</td><td>R@10</td><td>N@10</td><td>R@10</td><td>N@10</td></tr><tr><td>TIGER</td><td>0.0501</td><td>0.0242</td><td>0.0677</td><td>0.0342</td><td>0.0469</td><td>0.0228</td></tr><tr><td>LC-Rec</td><td>0.0527</td><td>0.0266</td><td>0.0694</td><td>0.0358</td><td>0.0494</td><td>0.0246</td></tr><tr><td>LETTER</td><td>0.0515</td><td>0.0257</td><td>0.0717</td><td>0.0375</td><td>0.0510</td><td>0.0265</td></tr><tr><td>UniTok</td><td>0.0592</td><td>0.0288</td><td>0.0835</td><td>0.0442</td><td>0.0591</td><td>0.0298</td></tr><tr><td>Gain</td><td>12.33%</td><td>8.27%</td><td>16.46%</td><td>17.87%</td><td>15.88%</td><td>12.45%</td></tr></table>

Table 4: Performance comparison among UniTok and recommendation competitors for the three unseen datasets. Here, the best and second-best performers are highlighted by bold and underline, respectively.

<table><tr><td></td><td colspan="2">Beauty</td><td colspan="2">Cellphones</td><td colspan="2">Grocery</td></tr><tr><td>Method</td><td>R@10</td><td>N@10</td><td>R@10</td><td>N@10</td><td>R@10</td><td>N@10</td></tr><tr><td>UniTok-1</td><td>0.0558</td><td>0.0304</td><td>0.0702</td><td>0.0371</td><td>0.0633</td><td>0.0342</td></tr><tr><td>UniTok-2</td><td>0.0896</td><td>0.0436</td><td>0.1194</td><td>0.0606</td><td>0.0989</td><td>0.0497</td></tr><tr><td>UniTok-3</td><td>0.0915</td><td>0.0457</td><td>0.1225</td><td>0.0622</td><td>0.1044</td><td>0.0515</td></tr><tr><td>UniTok</td><td>0.0934</td><td>0.0478</td><td>0.1251</td><td>0.0647</td><td>0.1061</td><td>0.0533</td></tr></table>

Table 5: Ablation study results on the Beauty, Cellphones, and Grocery datasets.

## Conclusions and Outlook

We explored an open yet fundamental challenge in multi-domain LLM-based recommendations by building a unified item tokenization framework. To this end, we proposed UniTok, which integrates a customized MoE architecture with codebooks to generate semantically meaningful tokens across diverse domains. Experiments on wide-ranging datasets showed that UniTok (a) achieves up to 51.89% gains in NDCG@10, (b) reduces trainable parameters by \( {9.63} \times \) , and (c) is theoretically validated with respect to the effectiveness of its individual components. Future work includes extending UniTok into a general-purpose tokenization interface for foundation models in recommendation.

## Acknowledgments

This work was supported by the National Research Foundation of Korea (NRF) funded by Korea Government (MSIT) under Grants RS-2021-NR059723 and RS-2023-00220762.

## References

Belghazi, M. I.; Baratin, A.; Rajeswar, S.; Ozair, S.; Bengio, Y.; Hjelm, R. D.; and Courville, A. C. 2018. Mutual Information Neural Estimation. In ICML, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018, 530-539.

Dai, D.; Deng, C.; Zhao, C.; Xu, R.; Gao, H.; Chen, D.; Li, J.; Zeng, W.; Yu, X.; Wu, Y.; et al. 2024. DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models. arXiv preprint arXiv:2401.06066.

Fedus, W.; Zoph, B.; and Shazeer, N. 2022. Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. Journal of Machine Learning Research, 23(120): 1-39.

Gretton, A.; Bousquet, O.; Smola, A.; and Schölkopf, B. 2005. Measuring Statistical Dependence with Hilbert-Schmidt Norms. In ALT, Singapore, October 8-11, 2005, 63-77.

Gururangan, S.; Marasovic, A.; Swayamdipta, S.; Lo, K.; Beltagy, I.; Downey, D.; and Smith, N. A. 2020. Don't Stop Pretraining: Adapt Language Models to Domains and Tasks. In ACL, Virtual Event, July 5-10, 2020, 8342-8360.

He, X.; Deng, K.; Wang, X.; Li, Y.; Zhang, Y.; and Wang, M. 2020. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. In SIGIR, Virtual Event, July 25-30, 2020, 639-648.

Hua, W.; Xu, S.; Ge, Y.; and Zhang, Y. 2023. How to Index Item IDs for Recommendation Foundation Models. In SIGIR-AP, Beijing, China, November 26-28, 2023, 195-204.

Jacobs, R. A.; Jordan, M. I.; Nowlan, S. J.; and Hinton, G. E. 1991. Adaptive Mixtures of Local Experts. Neural Computation, 3(1): 79-87.

Jain, Y.; Behl, H. S.; Kira, Z.; and Vineet, V. 2023. DAMEX: Dataset-aware Mixture-of-Experts for visual understanding of mixture-of-datasets. In NeurIPS, New Orleans, LA, USA, December 10 - 16, 2023.

Jiang, Y.; Li, Q.; Zhu, H.; Yu, J.; Li, J.; Xu, Z.; Dong, H.; and Zheng, B. 2022. Adaptive Domain Interest Network for Multi-Domain Recommendation. In CIKM, Atlanta, GA, USA, October 17-21, 2022, 3212-3221.

Kang, W.-C.; and McAuley, J. 2018. Self-Attentive Sequential Recommendation. In ICDM, Singapore, November 17- 20, 2018, 197-206.

Lepikhin, D.; Lee, H.; Xu, Y.; Chen, D.; Firat, O.; Huang, Y.; Krikun, M.; Shazeer, N.; and Chen, Z. 2021. GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding. In ICLR, Virtual Event, Austria, May 3-7, 2021.

Li, Y.; Pogodin, R.; Sutherland, D. J.; and Gretton, A. 2021. Self-Supervised Learning with Kernel Dependence Maximization. In NeurIPS, December 6-14, 2021, Virtual Event, 15543-15556.

Ma, R.; Tan, Y.; Zhou, X.; Chen, X.; Liang, D.; Wang, S.; Wu, W.; and Gui, T. 2022. Searching for Optimal Subword Tokenization in Cross-domain NER. In IJCAI, Vienna, Austria, 23-29 July 2022, 4289-4295.

Ning, W.; Yan, X.; Liu, W.; Cheng, R.; Zhang, R.; and Tang, B. 2023. Multi-Domain Recommendation with Embedding Disentangling and Domain Alignment. In CIKM, Birmingham, United Kingdom, October 21-25, 2023, 1917-1927.

Oord, A. v. d.; Li, Y.; and Vinyals, O. 2018. Representation Learning with Contrastive Predictive Coding. arXiv preprint arXiv:1807.03748.

Raffel, C.; Shazeer, N.; Roberts, A.; Lee, K.; Narang, S.; Matena, M.; Zhou, Y.; Li, W.; and Liu, P. J. 2020. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. Journal of Machine Learning Research, 21(140): 1-67.

Rajput, S.; Mehta, N.; Singh, A.; Keshavan, R. H.; Vu, T.; Heldt, L.; Hong, L.; Tay, Y.; Tran, V. Q.; Samost, J.; Kula, M.; Chi, E. H.; and Sathiamoorthy, M. 2023. Recommender Systems with Generative Retrieval. In NeurIPS, New Orleans, LA, USA, December 10 - 16, 2023.

Rendle, S.; Freudenthaler, C.; Gantner, Z.; and Schmidt-Thieme, L. 2009. BPR: Bayesian Personalized Ranking from Implicit Feedback. In UAI, Montreal, QC, Canada, June 18-21, 2009, 452-461.

Shazeer, N.; Mirhoseini, A.; Maziarz, K.; Davis, A.; Le, Q. V.; Hinton, G. E.; and Dean, J. 2017. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. In ICLR, Toulon, France, April 24-26, 2017.

Sun, F.; Liu, J.; Wu, J.; Pei, C.; Lin, X.; Ou, W.; and Jiang, P. 2019. BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. In CIKM, Beijing, China, November 3-7, 2019, 1441-1450.

Ullah, I.; Carrión-Ojeda, D.; Escalera, S.; Guyon, I.; Huis-man, M.; Mohr, F.; van Rijn, J. N.; Sun, H.; Vanschoren, J.; and Vu, P. A. 2022. Meta-Album: Multi-domain Meta-Dataset for Few-Shot Image Classification. In NeurIPS, New Orleans, LA, USA, November 28 - December 9, 2022.

Wang, W.; Bao, H.; Lin, X.; Zhang, J.; Li, Y.; Feng, F.; Ng, S.-K.; and Chua, T.-S. 2024. Learnable Item Tokenization for Generative Recommendation. In CIKM, Boise, ID, USA, October 21-25, 2024, 2400-2409.

Zhang, Y.; DING, H.; Shui, Z.; Ma, Y.; Zou, J.; Deoras, A.; and Wang, H. 2021. Language Models as Recommender Systems: Evaluations and Limitations. In I (Still) Can't Believe It's Not Better! NeurIPS 2021 Workshop.

Zheng, B.; Hou, Y.; Lu, H.; Chen, Y.; Zhao, W. X.; Chen, M.; and Wen, J.-R. 2024. Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation. In ICDE, Utrecht, The Netherlands, May 13-16, 2024, 1435- 1448.

# Appendix for "Tokenize Once, Recommend Anywhere: Unified Item Tokenization for Multi-domain LLM-based Recommendation"

List of contents in Appendix

- Summary of notations and their meanings used in the main manuscript

- Related work on LLM for generative recommendation and item tokenization: Appendix A

1 Detailed algorithm of training procedure for UniTok: Appendix B

- Formal proofs of the theorems in the main manuscript: Appendix C

- Additional experimental evaluations:

✓ Description and statistics of all datasets used: Appendix D. 1

✓ Overview of competitors for comparison: Appendix D. 2

✓ Formal definitions of evaluation metrics, including Recall@ \( M \) and NDCG@ \( M \) : Appendix D. 3

✓ Implementation details including hyperparameters and training setup: Appendix D.4

✓ Full performance results across all evaluation metrics: Appendix D. 5

✓ Ablation study for evaluation of contributions from individual components: Appendix D. 6

✓ Sensitivity analysis on robustness under different hyperparameter settings: Appendix D. 7

✓ Empirical validation for theorems: Appendix D. 8

Table of notations

<table><tr><td>Notation</td><td>Description</td></tr><tr><td>\( {\mathcal{D}}_{k} \)</td><td>The \( k \) -th dataset</td></tr><tr><td>\( K \)</td><td>The number of datasets</td></tr><tr><td>\( {\mathcal{I}}_{k} \)</td><td>Item set of \( {\mathcal{D}}_{k} \)</td></tr><tr><td>\( {\mathbf{X}}^{k} \)</td><td>Semantic embeddings for domain \( k \)</td></tr><tr><td>\( d \)</td><td>Embedding dimensionality</td></tr><tr><td>\( {\mathbf{x}}_{i}^{k} \)</td><td>Semantic embedding of the \( i \) -th item in \( {\mathcal{D}}_{k} \)</td></tr><tr><td>\( {f}_{\theta } \)</td><td>Shared encoder</td></tr><tr><td>\( {g}_{\phi } \)</td><td>Shared decoder</td></tr><tr><td>\( {\mathbf{Z}}^{k} \)</td><td>Hidden embeddings of \( {\mathbf{X}}^{k} \)</td></tr><tr><td>\( {\mathbf{z}}_{i}^{k} \)</td><td>Hidden embedding of \( {\mathbf{x}}_{i}^{k} \)</td></tr><tr><td>\( G\left( \cdot \right) \)</td><td>Router function</td></tr><tr><td>\( {G}_{k} \)</td><td>Routing probability assigned to the \( k \) -th expert</td></tr><tr><td>\( N \)</td><td>Number of selected top experts</td></tr><tr><td>\( {E}_{k} \)</td><td>The \( k \) -th expert module</td></tr><tr><td>\( {E}_{\text{ share }} \)</td><td>The shared expert module</td></tr><tr><td>\( L \)</td><td>Codebook size</td></tr><tr><td>\( T \)</td><td>Number of code vector in each codebook</td></tr><tr><td>\( {C}_{\ell } \)</td><td>The \( \ell \) -th codebook</td></tr><tr><td>\( {\mathbf{c}}_{i}^{k} \)</td><td>Discrete token of \( {\mathbf{x}}_{i}^{k} \)</td></tr><tr><td>\( \widehat{\operatorname{HSIC}}\left( {{\mathbf{X}}^{k},{\mathbf{Z}}^{k}}\right) \)</td><td>Empirical estimate of HSIC between \( {\mathbf{X}}^{k} \) and \( {\mathbf{Z}}^{k} \)</td></tr><tr><td>\( \mathbf{U},\mathbf{V} \)</td><td>Gaussian kernel matrices</td></tr><tr><td>\( {\widehat{I}}^{\left( k\right) } \)</td><td>\( \widehat{\operatorname{HSIC}}\left( {{\mathbf{X}}^{k},{\mathbf{Z}}^{k}}\right) \)</td></tr><tr><td>\( \operatorname{Var}\left\lbrack  {\widehat{I}}^{\left( k\right) }\right\rbrack \)</td><td>Variance of MI estimates across domains</td></tr><tr><td>\( \mathbb{E}{\left\lbrack  {\widehat{I}}^{\left( k\right) }\right\rbrack  }^{\tau } \)</td><td>Expectation of MI estimates across domains</td></tr><tr><td>\( {\mathcal{L}}_{\text{ Total }} \)</td><td>Total loss</td></tr><tr><td>\( {\mathcal{L}}_{\text{ Rec }} \)</td><td>Reconstruction loss</td></tr><tr><td>\( {\mathcal{L}}_{\mathrm{{RQ}}} \)</td><td>Resdual quantization loss</td></tr><tr><td>\( {\mathcal{L}}_{\mathrm{{MI}}} \)</td><td>Mutual information calibration loss</td></tr></table>

## A. Related Work

Our proposed method is related to two broader areas of research, namely LLM for generative recommendation and item tok-enization.

### 1.LLM for Generative Recommendation

LLMs have been widely adopted in recommender systems for their powerful generative capabilities (Li et al. 2024; Gong et al. 2023; Lin et al. 2024d). They have been utilized not only to enhance traditional recommendation methods but also to directly generate recommendations. A growing body of studies leveraged LLMs for data augmentation by synthesizing user-item interactions (Wei et al. 2024), incorporating external knowledge (Xi et al. 2024), and enriching user and item representations at the representation level (Liu et al. 2024; Qiu et al. 2021; Ren et al. 2024; Wu et al. 2022). LLM-based recommender systems employed a projection layer that aligns user and item embeddings within a shared latent space, enabling the computation of similarity scores for ranking tasks (Li et al. 2023a; Lin et al. 2024a; Zhang et al. 2025). Other methods focused on click-through rate prediction by feeding user profiles and item descriptions into LLMs to model their interactions, predicting the likelihood of user engagement (Bao et al. 2023; Lin et al. 2024b; Prakash et al. 2023). However, since LLMs originated from language processing, a gap exists between the language space and the item space when applying them to recommendation tasks.

## 2. Item Tokenization

In recommender systems powered by LLMs, a fundamental challenge arises from the gap between the language space and the item space. To address this issue, recent research has explored item tokenization strategies that effectively bridge this gap. Existing approaches can be broadly categorized into three types based on the nature of item identifiers. First, ID-based identifiers represent items as discrete tokens using unique IDs (Geng et al. 2022; Hua et al. 2023; Wang et al. 2024). While simple, these methods lack semantic information and generalization capability. Second, textual identifiers leverage item metadata, such as titles or descriptions, enabling LLMs to utilize pretrained linguistic knowledge for improved semantic understanding and generalization (Bao et al. 2025; Dai et al. 2023; Li et al. 2023b; Liao et al. 2023; Zhang et al. 2023, 2021; Lin et al. 2024c). Lastly, codebook-based identifiers employ learned discrete representations through vector quantization (Rajput et al. 2023; Wang et al. 2024; Zheng et al. 2024; Wang et al. 2024; Zheng et al. 2024; Zhai et al. 2025). These methods combine the interpretability of discrete tokens with the semantic richness of embeddings, providing structured and generalizable representations of items. However, common limitations of these item tokenization techniques hinder the generalization ability of LLMs. Specifically, most approaches rely on training separate models for each dataset (Rajput et al. 2023; Wang et al. 2024; Zheng et al. 2024; Wang et al. 2024; Zheng et al. 2024), while a more recent approach requires additional costly pre-training to transfer knowledge across domains for adaptation (Zhai et al. 2025).

## B. Algorithm of UniTok

Algorithm 1 outlines the training procedure of the proposed UniTok framework, which unifies item tokenization across multiple domains. Initially (Line 1), all core components, including the encoder, decoder, router, expert modules, and codebooks, are initialized. Each item \( {i}_{i}^{k} \) from domain \( {\mathcal{D}}_{k} \) is embedded into a semantic embedding \( {\mathbf{x}}_{i}^{k} \) using a pretrained embedding model (Lines 2-6). During training (Lines 7-13), mini-batches are sampled from mixed domains to encourage multi-domain generalization. Each input \( {\mathbf{x}}_{i}^{k} \) is first encoded into \( {\mathbf{z}}_{i}^{k} \) (Line 10), then passed through the TokenMoE module to obtain a quantized embedding \( {\widehat{\mathbf{z}}}_{i}^{k} \) and discrete token \( {\mathbf{c}}_{i}^{k} \) via expert routing and residual quantization (Line 11), and finally reconstructed \( {\widehat{\mathbf{x}}}_{i}^{k} \) via the decoder (Line 12). The total training objective combines a reconstruction loss (Line 14), a residual quantization loss (Line 15), and a mutual information calibration loss (Line 16) to enforce informativeness and semantic consistency. All parameters are updated jointly (Line 17), and the final output consists of discrete tokens for all items (Line 19).

## C. Theorems and Proofs

Theorem 1. The token space induced by UniTok exhibits strictly higher entropy than that of standard codebook-based methods

\[
H\left( {\mathcal{C}}_{\text{ UniTok }}\right)  > H\left( {\mathcal{C}}_{\text{ standard }}\right) ,
\]

where \( {\mathcal{C}}_{\text{ UniTok }} \) and \( {\mathcal{C}}_{\text{ standard }} \) denote the discrete token distributions generated by UniTok and standard codebook-based methods, respectively.

Proof. Consider the latent space representation of both models:

Entropy of standard codebook-based methods: In standard codebook-based methods, a single codebook with \( T \) code vectors is used at each of the \( L \) levels. The total number of unique tokens in the token space is

\[
\left| {\mathcal{C}}_{\text{ standard }}\right|  = {T}^{L}. \tag{15}
\]

Assuming a uniform probability distribution over the quantized representations:

\[
P\left( c\right)  = \frac{1}{{T}^{L}}, \tag{16}
\]

Algorithm 1: UniTok Training Procedure

---

Input: Multi-domain item datasets \( \left\{  {{\mathcal{D}}_{1},{\mathcal{D}}_{2},\ldots ,{\mathcal{D}}_{K}}\right\} \) with text metadata e.g., title, category, description)

Parameters: Encoder \( {f}_{\theta } \) , decoder \( {g}_{\phi } \) , router \( h\left( \cdot \right) \) , expert modules \( \left\{  {{E}_{1},\ldots ,{E}_{K},{E}_{\text{ share }}}\right\} \) with associated codebooks

Output: Discrete tokens \( {\mathbf{c}}_{i}^{k} \in  \mathcal{C} \) for all items

		Initialize all modules: \( {f}_{\theta },{g}_{\phi } \) , router \( h\left( \cdot \right) \) , experts \( {E}_{k} \) (including \( {E}_{\text{ share }} \) ), and their codebooks

		for \( k = 1 \) to \( K \) do

			for item \( {i}_{i}^{k} \in  {\mathcal{D}}_{k} \) do

				\( {\mathbf{x}}_{i}^{k} \leftarrow \) SemanticEmbedding \( \left( {i}_{i}^{k}\right) \)

			end for

		end for

		for epoch \( = 1 \) to \( E \) do

			Sample mini-batch \( \mathcal{B} = \left\{  {\mathbf{x}}_{i}^{k}\right\} \) from mixed domains

			for each item \( {\mathbf{x}}_{i}^{k} \) in batch \( \mathcal{B} \) do

				\( {\mathbf{z}}_{i}^{k} \leftarrow  {f}_{\theta }\left( {\mathbf{x}}_{i}^{k}\right) \) 																																													// Encode input

				\( {\widehat{\mathbf{z}}}_{i}^{k},{\mathbf{c}}_{i}^{k} \leftarrow  \operatorname{TokenMoE}\left( {\mathbf{z}}_{i}^{k}\right) \) 																																								// Discretize via TokenMoE

				\( {\widehat{\mathbf{x}}}_{i}^{k} \leftarrow  {g}_{\phi }\left( {\widehat{\mathbf{z}}}_{i}^{k}\right) \) 																																									// Decode reconstruction

			end for

			Compute reconstruction loss \( {\mathcal{L}}_{\text{ Rec }} \) in the main manuscript of Eq. (3)

			Compute RQ loss \( {\mathcal{L}}_{\mathrm{{RQ}}} \) in the main manuscript of Eq. (8)

			Compute MI loss \( {\mathcal{L}}_{\mathrm{{MI}}} \) in the main manuscript of Eq. (10)

			Update \( {f}_{\theta },{g}_{\phi }, h\left( \cdot \right) \) , and expert parameters

		end for

		return token assignments \( {\mathbf{c}}_{i}^{k} \) for all items

---

The entropy of the token space in standard codebook-based methods is

\[
H\left( {\mathcal{C}}_{\text{ standard }}\right)  =  - \mathop{\sum }\limits_{{c \in  \mathcal{C}}}P\left( c\right) \log P\left( c\right) . \tag{17}
\]

Substituting \( P\left( c\right) \) , we have

\[
H\left( {\mathcal{C}}_{\text{ standard }}\right)  = L\log T. \tag{18}
\]

Entropy of UniTok: In our UniTok, there are \( K \) domain-specific experts, each with its own independent codebook of size \( {T}^{L} \) . Given an item embedding \( \mathbf{z} \) , the router function assigns a probability distribution over experts:

\[
{G}_{k} = \frac{\exp \left( {h\left( {\mathbf{z}}^{\left( k\right) }\right) }\right) }{\mathop{\sum }\limits_{{j = 1}}^{K}\exp \left( {h\left( {\mathbf{z}}^{\left( j\right) }\right) }\right) }, \tag{19}
\]

where the \( h\left( \cdot \right) \) is the linear transformation and \( {\mathbf{z}}^{\left( k\right) } \) denotes the logit corresponding to the \( k \) -th domain-specific expert.

Let \( \mathcal{G} \) denote the random variable representing expert selection. An item token \( c \) in \( {\mathcal{C}}_{\text{ UniTok }} \) is then generated by

- Sampling an expert index \( k \sim  \mathcal{G} \) ,

- Sampling a token \( c \sim  {\mathcal{C}}_{k} \) .

Here, \( {\mathcal{C}}_{k} \) is the discrete token distributions in the \( k \) -th expert.

By the chain rule of entropy, it follows that

\[
H\left( {\mathcal{C}}_{\text{ UniTok }}\right)  = H\left( \mathcal{G}\right)  + H\left( {{\mathcal{C}}_{k} \mid  \mathcal{G}}\right) . \tag{20}
\]

The conditional entropy term \( H\left( {{\mathcal{C}}_{k} \mid  \mathcal{G}}\right) \) is the expectation of the entropy of each expert’s latent space, weighted by the distribution from the router, and is written as

\[
H\left( {{\mathcal{C}}_{k} \mid  \mathcal{G}}\right)  = \mathop{\sum }\limits_{{k = 1}}^{K}{G}_{k} \cdot  H\left( {\mathcal{C}}_{k}\right)  = {\mathbb{E}}_{k \sim  \mathcal{G}}\left\lbrack  {H\left( {\mathcal{C}}_{k}\right) }\right\rbrack  . \tag{21}
\]

Then, we have

\[
H\left( {\mathcal{C}}_{\text{ UniTok }}\right)  = H\left( \mathcal{G}\right)  + {\mathbb{E}}_{k \sim  \mathcal{G}}\left\lbrack  {H\left( {\mathcal{C}}_{k}\right) }\right\rbrack \tag{22}
\]

Since each expert has the same quantization entropy as standard codebook-based methods, it follows that

\[
H\left( {\mathcal{C}}_{k}\right)  = L\log T\text{ for all }k, \tag{23}
\]

We then obtain

\[
{\mathbb{E}}_{k \sim  \mathcal{G}}\left\lbrack  {H\left( {\mathcal{C}}_{k}\right) }\right\rbrack   = L\log T \tag{24}
\]

The router entropy \( H\left( \mathcal{G}\right) \) is

\[
H\left( \mathcal{G}\right)  =  - \mathop{\sum }\limits_{{k = 1}}^{K}{G}_{k}\log {G}_{k} \tag{25}
\]

which is always positive when \( K > 1 \) , i.e., \( H\left( \mathcal{G}\right)  > 0 \) .

Thus, the entropy of \( {\mathcal{C}}_{\text{ UniTok }} \) is

\[
H\left( {\mathcal{C}}_{\text{ UniTok }}\right)  = H\left( \mathcal{G}\right)  + L\log T > H\left( {\mathcal{C}}_{\text{ standard }}\right) . \tag{26}
\]

This completes the proof of Theorem 1.

Before proving Theorem 2, we begin with Lemma 1 below.

Lemma 1. The function \( f\left( \mathbf{x}\right)  = \parallel \mathbf{x}{\parallel }^{2} = {\mathbf{x}}^{\top }\mathbf{x} \) , where \( \mathbf{x} \in  {\mathbb{R}}^{n} \) , is convex.

Proof. To show that \( f\left( \mathbf{x}\right) \) is convex, we check whether its Hessian matrix is positive semidefinite.

Step 1: Compute the gradient of \( f\left( \mathbf{x}\right) \) as follows:

\[
\nabla f\left( \mathbf{x}\right)  = \nabla \left( {{\mathbf{x}}^{\top }\mathbf{x}}\right)  = 2\mathbf{x}. \tag{27}
\]

Step 2: Compute the Hessian of \( f\left( \mathbf{x}\right) \) as follows:

\[
{\nabla }^{2}f\left( \mathbf{x}\right)  = \nabla \left( {2\mathbf{x}}\right)  = 2\mathbf{I}, \tag{28}
\]

where \( \mathbf{I} \) is the \( n \times  n \) identity matrix.

Therefore, the Hessian is positive definite, which implies that \( f\left( \mathbf{x}\right) \) is a strictly convex function. This completes the proof of Lemma 1.

Theorem 2. Let \( \mathbb{E}\left\lbrack  {\mathcal{L}}_{\text{ UniTok }}\right\rbrack \) and \( \mathbb{E}\left\lbrack  {\mathcal{L}}_{\text{ standard }}\right\rbrack \) denote the expected quantization error of UniTok and standard codebook-based methods using a single shared codebook, respectively. Then, the following inequality holds:

\[
\mathbb{E}\left\lbrack  {\mathcal{L}}_{\text{ UniTok }}\right\rbrack   \leq  \mathbb{E}\left\lbrack  {\mathcal{L}}_{\text{ standard }}\right\rbrack  .
\]

Proof. Let \( \mathbf{z} \in  {\mathbb{R}}^{d} \) be an item latent embedding drawn from a distribution \( p\left( z\right) \) . Let \( \widehat{\mathbf{z}} = E\left( \mathbf{z}\right) \) denote the output of the quantization function, which generates a combination of code vectors to approximate the item latent embedding \( \mathbf{z} \) .

For standard codebook-based methods using a single set of codebooks, the expected quantization error is given by

\[
\mathbb{E}\left\lbrack  {\mathcal{L}}_{\text{ standard }}\right\rbrack   = {\mathbb{E}}_{\mathbf{z} \sim  p\left( z\right) }\left\lbrack  {\parallel \mathbf{z} - E\left( \mathbf{z}\right) {\parallel }^{2}}\right\rbrack  . \tag{29}
\]

Now, consider UniTok with \( K \) domain-specific experts, each with its own set of codebooks. Given the embedding \( \mathbf{z} \) , the router function that assigns probabilities to each expert:

\[
{G}_{k} = \frac{\exp \left( {h\left( {\mathbf{z}}^{\left( k\right) }\right) }\right) }{\mathop{\sum }\limits_{{j = 1}}^{K}\exp \left( {h\left( {\mathbf{z}}^{\left( j\right) }\right) }\right) }. \tag{30}
\]

The approximated latent embedding is generated by \( \widehat{\mathbf{z}} = \mathop{\sum }\limits_{{k = 1}}^{K}{G}_{k} \cdot  {E}_{k}\left( \mathbf{z}\right) \) . Thus, the quantization error is

\[
{\mathcal{L}}_{\text{ UniTok }}\left( \mathbf{z}\right)  = {\begin{Vmatrix}\mathbf{z} - \mathop{\sum }\limits_{{k = 1}}^{K}{G}_{k} \cdot  {E}_{k}\left( \mathbf{z}\right) \end{Vmatrix}}^{2}. \tag{31}
\]

where \( {E}_{k}\left( \mathbf{z}\right) \) is the quantization function of the \( k \) -th expert module. As \( \mathop{\sum }\limits_{{k = 1}}^{K}{G}_{k} = 1 \) , we have

\[
{\mathcal{L}}_{\text{ UniTok }}\left( \mathbf{z}\right)  = {\begin{Vmatrix}\mathop{\sum }\limits_{{k = 1}}^{K}{G}_{k} \cdot  \left( \mathbf{z} - {E}_{k}\left( \mathbf{z}\right) \right) \end{Vmatrix}}^{2}. \tag{32}
\]

We define the approximation error for each \( z \) as

\[
{\epsilon }_{k}\left( \mathbf{z}\right)  = \mathbf{z} - {E}_{k}\left( \mathbf{z}\right) . \tag{33}
\]

Replacing with \( {\epsilon }_{k}\left( \mathbf{z}\right) \) , we have

\[
{\mathcal{L}}_{\text{ UniTok }}\left( \mathbf{z}\right)  = {\begin{Vmatrix}\mathop{\sum }\limits_{{k = 1}}^{K}{G}_{k} \cdot  {\epsilon }_{k}\left( \mathbf{z}\right) \end{Vmatrix}}^{2}. \tag{34}
\]

Using Jensen's inequality, we apply the convexity of the squared norm function based on Lemma 1:

\[
{\mathcal{L}}_{\text{ UniTok }}\left( \mathbf{z}\right)  = {\begin{Vmatrix}\mathop{\sum }\limits_{{k = 1}}^{K}{G}_{k} \cdot  {\epsilon }_{k}\left( \mathbf{z}\right) \end{Vmatrix}}^{2} \leq  \mathop{\sum }\limits_{{k = 1}}^{K}{G}_{k} \cdot  {\begin{Vmatrix}{\epsilon }_{k}\left( \mathbf{z}\right) \end{Vmatrix}}^{2}. \tag{35}
\]

Taking the expectation over the data distribution \( p\left( z\right) \) on both sides of Eq. (21), we obtain

\[
{\mathbb{E}}_{\mathbf{z} \sim  p\left( z\right) }\left\lbrack  {\begin{Vmatrix}\mathop{\sum }\limits_{{k = 1}}^{K}{G}_{k} \cdot  {\epsilon }_{k}\left( \mathbf{z}\right) \end{Vmatrix}}^{2}\right\rbrack   \leq  {\mathbb{E}}_{\mathbf{z} \sim  p\left( z\right) }\left\lbrack  {\mathop{\sum }\limits_{{k = 1}}^{K}{G}_{k} \cdot  {\begin{Vmatrix}{\epsilon }_{k}\left( \mathbf{z}\right) \end{Vmatrix}}^{2}}\right\rbrack  . \tag{36}
\]

Since the single quantization function \( E\left( \mathbf{z}\right) \) can be viewed as a degenerate case where \( {G}_{k} = 1 \) for a single expert and 0 for all others, the UniTok always provides a better or equal approximation:

\[
{\mathbb{E}}_{\mathbf{z} \sim  p\left( z\right) }\left\lbrack  {{\mathcal{L}}_{\text{ UniTok }}\left( \mathbf{z}\right) }\right\rbrack   \leq  {\mathbb{E}}_{\mathbf{z} \sim  p\left( z\right) }\left\lbrack  {{\mathcal{L}}_{\text{ standard }}\left( \mathbf{z}\right) }\right\rbrack  . \tag{37}
\]

This completes the proof of Theorem 2.

Theorem 3. Suppose that the loss \( {\mathcal{L}}^{\left( k\right) } \) on the \( k \) -th domain is Lipschitz-continuous with respect to the informativeness of representations. Then, the performance variability across domains is upper-bounded by the variance of MI:

\[
\left| {{\mathcal{L}}^{\left( i\right) } - {\mathcal{L}}^{\left( j\right) }}\right|  \leq  C\sqrt{\operatorname{Var}\left\lbrack  {\widehat{I}}^{\left( k\right) }\right\rbrack  },\forall i, j
\]

where \( \operatorname{Var}\left\lbrack  {\widehat{I}}^{\left( k\right) }\right\rbrack \) is the variance of MI estimates across domains and \( C \) is a constant.

Proof. Let \( {\widehat{I}}^{\left( k\right) } = \widehat{\operatorname{HSIC}}\left( {{\mathbf{X}}^{k},{\mathbf{Z}}^{k}}\right) \) be the MI between the input \( {\mathbf{X}}^{k} \) and its representation \( {\mathbf{Z}}^{k} \) for the \( k \) -th domain. The loss \( {\mathcal{L}}^{\left( k\right) } \) on the \( k \) -th domain is Lipschitz-continuous with respect to the informativeness of representation, i.e.,

\[
\left| {{\mathcal{L}}^{\left( i\right) } - {\mathcal{L}}^{\left( j\right) }}\right|  \leq  L\left| {\widehat{\operatorname{HSIC}}\left( {{\mathbf{X}}^{i};{\mathbf{Z}}^{i}}\right)  - \widehat{\operatorname{HSIC}}\left( {{\mathbf{X}}^{j};{\mathbf{Z}}^{j}}\right) }\right| , \tag{38}
\]

where \( L \) is the Lipschitz constant that quantifies the sensitivity of the downstream loss to changes in representation informativeness (i.e., MI).

This assumption ensures that variations in representation informativeness (measured by MI) translate into bounded variations in the downstream task loss, effectively connecting the input-output MI stability (?). As a result, controlling the variance of MI across domains allows us to stabilize performance fluctuations.

Now, we take the maximum gap in MI across all domain pairs and relate it to the variance of MI:

\[
\left| {{\widehat{I}}^{\left( i\right) } - {\widehat{I}}^{\left( j\right) }}\right|
\]

\[
\leq  \mathop{\max }\limits_{{i, j}}\left| {{\widehat{I}}^{\left( i\right) } - {\widehat{I}}^{\left( j\right) }}\right|
\]

\[
= \mathop{\max }\limits_{{i, j}}\left| {{\widehat{I}}^{\left( i\right) } - \mu  - \left( {{\widehat{I}}^{\left( j\right) } - \mu }\right) }\right|
\]

\[
\leq  \mathop{\max }\limits_{{i, j}}\left( {\left| {{\widehat{I}}^{\left( i\right) } - \mu }\right|  + \left| \left( {{\widehat{I}}^{\left( j\right) } - \mu }\right) \right| }\right) \tag{39}
\]

\[
\leq  2\mathop{\max }\limits_{k}\left| {{\widehat{I}}^{\left( k\right) } - \mu }\right|
\]

\[
\leq  2\sqrt{\mathop{\sum }\limits_{{k = 1}}^{K}{\left( {\widehat{I}}^{\left( k\right) } - \mu \right) }^{2}}
\]

\[
\leq  2\sqrt{{KVar}\left\lbrack  {\widehat{I}}^{\left( k\right) }\right\rbrack  }.
\]

Therefore, we obtain

\[
\left| {{\mathcal{L}}^{\left( i\right) } - {\mathcal{L}}^{\left( j\right) }}\right|  \leq  L\left| {{\widehat{I}}^{\left( i\right) } - {\widehat{I}}^{\left( j\right) }}\right|  \leq  C\sqrt{\operatorname{Var}\left\lbrack  {\widehat{I}}^{\left( k\right) }\right\rbrack  },
\]

where \( C = {2L}\sqrt{K} \) .

This completes the proof of Theorem 3.

## D. Additional Experimental Evaluations

## 1. Datasets

We conduct our experiments on ten real-world datasets widely adopted for evaluating the performance of recommendations, which include Beauty, Cellphones, Grocery, Instruments, Office, Pet Supplies, Tools, Toys, Games \( {}^{6} \) , and Yelp \( {}^{7} \) . Each item in these datasets contains rich content information, including title, category, and description. Additionally, we use three datasets, Clothing, Health, and Sports \( {}^{8} \) , for zero-shot evaluation. Table 6 provides a summary of the statistics for all datasets.

---

\( {}^{6} \) https://nijianmo.github.io/amazon/index.html.

\( {}^{7} \) https://www.yelp.com/dataset.

---

<table><tr><td>Dataset</td><td>#of users</td><td>#of items</td><td>#of interactions</td></tr><tr><td>Beauty</td><td>22,363</td><td>12,101</td><td>198,502</td></tr><tr><td>Cellphones</td><td>27,879</td><td>10,429</td><td>194,439</td></tr><tr><td>Grocery</td><td>14,681</td><td>8,713</td><td>151,254</td></tr><tr><td>Instruments</td><td>24,772</td><td>9,922</td><td>206,153</td></tr><tr><td>Office Products</td><td>4,905</td><td>2,420</td><td>53,258</td></tr><tr><td>Pet Supplies</td><td>19,856</td><td>8,510</td><td>157,836</td></tr><tr><td>Tools</td><td>16,638</td><td>10,217</td><td>134,476</td></tr><tr><td>Toys</td><td>19,412</td><td>11,924</td><td>167,597</td></tr><tr><td>Games</td><td>24,303</td><td>10,672</td><td>231,780</td></tr><tr><td>Yelp</td><td>30,431</td><td>20,033</td><td>316,354</td></tr><tr><td>Clothing</td><td>39,387</td><td>23,033</td><td>278,677</td></tr><tr><td>Health</td><td>38609</td><td>18533</td><td>346,355</td></tr><tr><td>Sports</td><td>35,598</td><td>18,357</td><td>296,337</td></tr></table>

Table 6: The statistics of datasets from ten domains suited to recommendations.

## 2. Competitors

To comprehensively demonstrate the superiority of UniTok, we present nine benchmark recommendation methods, including four widely-used collaborative filtering methods, MF (Rendle et al. 2009), SASRec (Kang and McAuley 2018), LightGCN (He et al. 2020), Bert4Rec (Sun et al. 2019), and five item tokenization-aided recommendations, P5-TID (Hua et al. 2023), P5-SemID (Hua et al. 2023), TIGER (Rajput et al. 2023), LC-Rec (Zheng et al. 2024), and LETTER (Wang et al. 2024). We implemented all these methods using the parameter settings described in their original articles.

- MF (Rendle et al. 2009). This models user-item interactions by decomposing them into latent user and item embeddings through matrix factorization.

- SASRec (Kang and McAuley 2018). This method uses self-attention to model user sequences, effectively capturing both short- and long-term dependencies with high efficiency.

- LightGCN (He et al. 2020). This model simplifies graph convolutional networks (GCNs) for recommendation by focusing solely on neighborhood aggregation, eliminating feature transformation and nonlinear activation to enhance performance.

- Bert4Rec (Sun et al. 2019). This method leverages bidirectional self-attention with a Cloze-style objective to model user behavior sequences, enabling richer context modeling than unidirectional sequential methods.

- P5-TID (Hua et al. 2023). This method uses item titles as textual identifiers to enable LLM-based generative recommendation.

- P5-SemID (Hua et al. 2023). This method constructs item identifiers from metadata (e.g., attributes) for LLM-based generative recommendation.

- TIGER (Rajput et al. 2023). This method trains a sequence-to-sequence model to autoregressively generate item identifiers composed of semantic codeword tuples for next-item prediction.

- LC-Rec (Zheng et al. 2024). This method integrates language and collaborative semantics by learning meaningful item indices via vector quantization and tuning LLMs through alignment tasks for direct item generation.

- LETTER (Wang et al. 2024). This method is a learnable tokenizer for LLM-based generative recommendation that combines residual quantization, contrastive alignment, and diversity loss to encode hierarchical semantics, capture collaborative signals, and mitigate code assignment bias.

## 3. Performance Metrics

We follow the full-ranking evaluation protocol (He et al. 2020), where all non-interacted items are ranked for each user. To evaluate performance, we adopt two widely used ranking metrics: Recall@M (R@M) and NDCG@M (N@M), where \( M \in  \{ 5,{10}\} \) . We formally define the ranking metrics used in our experiments.

Recall@M.Recall@M measures the proportion of relevant items successfully retrieved in the top- \( M \) recommendations:

---

\[
\text{ Recall@M } = \frac{\left| \text{ Top-M recommended items } \cap  \text{ Ground truth ttems }\right| }{\left| \text{ Ground truth items }\right| }.
\]

\( {}^{8} \) https://nijianmo.github.io/amazon/index.html.

---

NDCG@M (Normalized Discounted Cumulative Gain). NDCG@M considers both the relevance and the ranking positions of the recommended items:

\[
\mathrm{{NDCG}}@M = \frac{1}{\mathrm{{IDCG}}@M}\mathop{\sum }\limits_{{i = 1}}^{M}\frac{\mathbb{1}\left\{  {{\text{ item }}_{i} \in  \text{ Ground truth }}\right\}  }{{\log }_{2}\left( {i + 1}\right) },
\]

where IDCG@ \( M \) denotes the ideal DCG, i.e., the maximum possible DCG for the given ground truth, ensuring normalization between 0 and 1.

## 4. Implementation Details

To evaluate recommendation performance, we adopt TIGER (Rajput et al. 2023), a representative LLM-based generative recommender system that uses T5 (Raffel et al. 2020) as its backbone. We follow this design and retain T5 in our experiments due to its unified text-to-text framework, strong performance across a wide range of generative tasks, and efficiency in conditional sequence generation-making it well-suited to the generative recommendation paradigm adopted by TIGER.

<table><tr><td></td><td colspan="2">Beauty</td><td colspan="2">Cellphones</td><td colspan="2">Grocery</td><td colspan="2">Insturments</td><td colspan="2">Office Products</td></tr><tr><td>Method</td><td>R@10</td><td>N@10</td><td>R@10</td><td>N@10</td><td>R@10</td><td>N@10</td><td>R@10</td><td>N@10</td><td>R@10</td><td>N@10</td></tr><tr><td>MF</td><td>0.0614</td><td>0.0369</td><td>0.0604</td><td>0.0267</td><td>0.0418</td><td>0.0216</td><td>0.0930</td><td>0.0710</td><td>0.0569</td><td>0.0255</td></tr><tr><td>LightGCN</td><td>0.0639</td><td>0.0285</td><td>0.0668</td><td>0.0456</td><td>0.0697</td><td>0.0357</td><td>0.1008</td><td>0.0781</td><td>0.0587</td><td>0.0301</td></tr><tr><td>SASRec</td><td>0.0646</td><td>0.0314</td><td>0.0651</td><td>0.0446</td><td>0.0701</td><td>0.0376</td><td>0.0905</td><td>0.0609</td><td>0.0574</td><td>0.0285</td></tr><tr><td>Bert4Rec</td><td>0.0372</td><td>0.0194</td><td>0.0507</td><td>0.0268</td><td>0.0448</td><td>0.0237</td><td>0.0791</td><td>0.0573</td><td>0.0563</td><td>0.0274</td></tr><tr><td>P5-TID</td><td>0.0532</td><td>0.0255</td><td>0.0648</td><td>0.0357</td><td>0.0617</td><td>0.0316</td><td>0.0928</td><td>0.0721</td><td>0.0557</td><td>0.0239</td></tr><tr><td>P5-SemID</td><td>0.0584</td><td>0.0304</td><td>0.0737</td><td>0.0406</td><td>0.0641</td><td>0.0351</td><td>0.0964</td><td>0.0730</td><td>0.0592</td><td>0.0283</td></tr><tr><td>TIGER</td><td>0.0624</td><td>0.0324</td><td>0.0838</td><td>0.0446</td><td>0.0706</td><td>0.0375</td><td>0.1047</td><td>0.0788</td><td>0.0594</td><td>0.0295</td></tr><tr><td>LC-Rec</td><td>0.0684</td><td>0.0381</td><td>0.0859</td><td>0.0458</td><td>0.0722</td><td>0.0369</td><td>0.1066</td><td>0.0802</td><td>0.0637</td><td>0.0311</td></tr><tr><td>LETTER</td><td>0.0672</td><td>0.0364</td><td>0.0876</td><td>0.0473</td><td>0.0731</td><td>0.0392</td><td>0.1122</td><td>0.0831</td><td>0.0649</td><td>0.0326</td></tr><tr><td>UniTok</td><td>0.0934</td><td>0.0478</td><td>0.1251</td><td>0.0647</td><td>0.1061</td><td>0.0533</td><td>0.1361</td><td>0.0884</td><td>0.0897</td><td>0.0432</td></tr><tr><td>Improve</td><td>36.55%</td><td>25.46%</td><td>42.81%</td><td>36.78%</td><td>45.14%</td><td>35.97%</td><td>21.30%</td><td>6.38%</td><td>38.21%</td><td>32.52%</td></tr><tr><td></td><td colspan="2">Pet Supplies</td><td colspan="2">Tools</td><td colspan="2">Toys</td><td colspan="2">Games</td><td colspan="2">Yelp</td></tr><tr><td>Method</td><td>R@10</td><td>N@10</td><td>R@10</td><td>N@10</td><td>R@10</td><td>N@10</td><td>R@10</td><td>N@10</td><td>R@10</td><td>N@10</td></tr><tr><td>MF</td><td>0.0503</td><td>0.0268</td><td>0.0356</td><td>0.0169</td><td>0.0405</td><td>0.0192</td><td>0.0359</td><td>0.0366</td><td>0.0304</td><td>0.0144</td></tr><tr><td>LightGCN</td><td>0.0529</td><td>0.0289</td><td>0.0482</td><td>0.0257</td><td>0.0495</td><td>0.0287</td><td>0.0407</td><td>0.0417</td><td>0.0368</td><td>0.0195</td></tr><tr><td>SASRec</td><td>0.0538</td><td>0.0301</td><td>0.0475</td><td>0.0234</td><td>0.0473</td><td>0.0239</td><td>0.0401</td><td>0.0412</td><td>0.0354</td><td>0.0183</td></tr><tr><td>Bert4Rec</td><td>0.0313</td><td>0.0161</td><td>0.0184</td><td>0.0092</td><td>0.0333</td><td>0.0177</td><td>0.0363</td><td>0.0379</td><td>0.0272</td><td>0.0131</td></tr><tr><td>P5-TID</td><td>0.0485</td><td>0.0243</td><td>0.0377</td><td>0.0198</td><td>0.0419</td><td>0.0202</td><td>0.0372</td><td>0.0388</td><td>0.0316</td><td>0.0154</td></tr><tr><td>P5-SemID</td><td>0.0569</td><td>0.0282</td><td>0.0405</td><td>0.0237</td><td>0.0445</td><td>0.0231</td><td>0.0398</td><td>0.0432</td><td>0.0324</td><td>0.0188</td></tr><tr><td>TIGER</td><td>0.0546</td><td>0.0279</td><td>0.0507</td><td>0.0284</td><td>0.0486</td><td>0.0268</td><td>0.0438</td><td>0.0427</td><td>0.0394</td><td>0.0208</td></tr><tr><td>LC-Rec</td><td>0.0648</td><td>0.0335</td><td>0.0561</td><td>0.0307</td><td>0.0538</td><td>0.0279</td><td>0.0442</td><td>0.0451</td><td>0.0418</td><td>0.0215</td></tr><tr><td>LETTER</td><td>0.0596</td><td>0.0307</td><td>0.0556</td><td>0.0298</td><td>0.0546</td><td>0.0291</td><td>0.0559</td><td>0.0469</td><td>0.0426</td><td>0.0231</td></tr><tr><td>UniTok</td><td>0.0955</td><td>0.0496</td><td>0.0852</td><td>0.0439</td><td>0.0902</td><td>0.0442</td><td>0.0565</td><td>0.0476</td><td>0.0684</td><td>0.0321</td></tr><tr><td>Improve</td><td>47.38%</td><td>48.06%</td><td>51.87%</td><td>42.99%</td><td>65.20%</td><td>51.89%</td><td>1.07%</td><td>1.49%</td><td>60.56%</td><td>38.96%</td></tr></table>

Table 7: Performance comparison among UniTok and recommendation competitors for the ten benchmark datasets. Here, the best and second-best performers are highlighted by bold and underline, respectively. The improvements are statistically significant on average \( \left( {p = {0.0219} < {0.05}}\right) \) based on paired t-tests over five runs across all datasets.

While other LLMs (e.g., GPT, BART, and LLaMA) are viable options, our focus is to ensure a fair and controlled evaluation of our proposed tokenization framework, UniTok. Using T5, consistent with other codebook based methods like TIGER (Rajput et al. 2023) and LETTER (Wang et al. 2024), allows us to attribute performance improvements to our method rather than differences in model architecture. Additionally, T5 offers a favorable balance between capability and computational efficiency, which is important for reproducible experimentation. We leave the exploration of other LLMs as promising future work.

For item tokenization, we use 4-level codebook-based identifiers, where each codebook comprises 256 code vectors with an embedding dimension of 32. To obtain item semantic embeddings, we follow (Rajput et al. 2023; Wang et al. 2024) and use a pre-trained language model to encode item content information. The resulting semantic embedding has a dimensionality of 2048, while the hidden embedding dimension is 32. The shared expert is initialized with the average item embedding across all domains, capturing general semantic patterns over domains, whereas the \( K = {10} \) domain experts-corresponding to ten domains-are initialized with their respective domain-specific average embeddings, naturally inducing domain-aware assignment without supervision. Following (Rajput et al. 2023; Wang et al. 2024), we set \( \alpha \) as 0.25 and \( \beta \) as 1 . And \( {\lambda }_{\mathrm{{RO}}} \) is typically set to 1 and \( {\lambda }_{\mathrm{{MI}}} \) in the range of \( \{ {0.01},{0.03},{0.05},{0.07},{0.1}\} \) . We train UniTok for 10k epochs via AdamW [27] optimizer with a learning rate of \( {1e} - 3 \) and a batch size of1,024. TIGER is fine-tuned for convergence based on the validation performance, with a learning rate in 1e-3, 5e-4 and 1e-4, 2e-4, 3e-4.

All experiments are carried out with Intel (R) 12-Core (TM) E5-1650 v4 CPUs @ 3.60GHz and NVIDIA GeForce RTX 3090 GPUs.

## 5. Complete Set of Experimental Results

To evaluate the recommendation accuracy of UniTok, we compare it with baseline tokenization methods across multiple benchmark datasets, measuring performance using both Recall@10 and NDCG@10 (note that we have shown the results only with respect to NDCG@10 in the main manuscript due to space limit). Unlike existing approaches that train a separate tokenizer per dataset, UniTok is trained once across all domains. As shown in Table 7, UniTok consistently outperforms competitors, achieving up to 65.20% improvement in Recall @ 10 on the Toys dataset, as confirmed by paired t-tests showing statistically significant improvements ( \( p < {0.05} \) ). The results demonstrate the effectiveness of our unified tokenization framework; its shared tokenization captures item semantics across diverse domains without domain-specific customization, highlighting the model's scalability and generality.

LLM-based recommender systems using item tokenization, such as TIGER, LC-Rec, LETTER, and UniTok, consistently outperform traditional collaborative filtering methods (e.g., MF, LightGCN, SASRec, and Bert4Rec), benefiting from LLMs' semantic reasoning capabilities. Moreover, incorporating learned tokenization yields further gains over metadata-based approaches (e.g., P5-TID and P5-SemID), as tokenization bridges the item and language domains, enabling discrete, semantically rich item representations that enhance generalization.

## 6. Ablation Study

To assess the contribution of each component in UniTok, we perform an ablation study by progressively removing or modifying its core modules: the TokenMoE module, the shared expert, and the MI calibration part. UniTok-1 removes the TokenMoE and MI calibration; UniTok-2 keeps TokenMoE, but removes the shared expert and MI calibration; and UniTok-3 only removes the MI calibration (note that we have shown the results only on three datasets in the main manuscript due to space limit). UniTok includes all components. As shown in Table 8, removing any module leads to a noticeable drop in recommendation accuracy, which confirms that the combination of modules is crucial to UniTok's effectiveness. In particular, comparing UniTok-1 and UniTok-2 highlights the importance of the TokenMoE module, which significantly improves performance across all datasets by capturing dataset-specific token semantics. Additionally, in comparison with UniTok-3, introducing MI calibration further enhances performance by encouraging the learned representations to retain essential semantic information from the original input space. For clarity and focus, we report results on five representative domains that exhibit trends consistent with those observed across the full range of evaluated datasets.

<table><tr><td></td><td colspan="2">Beauty</td><td colspan="2">Cellphones</td><td colspan="2">Grocery</td><td colspan="2">Instruments</td><td colspan="2">Yelp</td></tr><tr><td>Method</td><td>R@10</td><td>N@10</td><td>R@10</td><td>N@10</td><td>R@10</td><td>N@10</td><td>N@10</td><td>R@10</td><td>N@10</td><td>R@10</td></tr><tr><td>UniTok-1</td><td>0.0558</td><td>0.0304</td><td>0.0702</td><td>0.0371</td><td>0.0633</td><td>0.0342</td><td>0.0926</td><td>0.0742</td><td>0.0345</td><td>0.0177</td></tr><tr><td>UniTok-2</td><td>0.0896</td><td>0.0436</td><td>0.1194</td><td>0.0606</td><td>0.0989</td><td>0.0497</td><td>0.1273</td><td>0.0851</td><td>0.0624</td><td>0.0281</td></tr><tr><td>UniTok-3</td><td>0.0915</td><td>0.0457</td><td>0.1225</td><td>0.0622</td><td>0.1044</td><td>0.0515</td><td>0.1327</td><td>0.0868</td><td>0.0657</td><td>0.0303</td></tr><tr><td>UniTok</td><td>0.0934</td><td>0.0478</td><td>0.1251</td><td>0.0647</td><td>0.1061</td><td>0.0533</td><td>0.1361</td><td>0.0884</td><td>0.0684</td><td>0.0321</td></tr></table>

Table 8: Ablation study results on the Beauty, Cellphones, Grocery, Instrument, and Yelp datasets.

### 7.How Sensitive is UniTok to Key Parameters?

We analyze the sensitivity of UniTok to key hyperparameters, including the number of quantization levels \( L \) , codebook size \( T \) , and the loss weights \( {\lambda }_{\mathrm{{RO}}} \) and \( {\lambda }_{\mathrm{{MI}}} \) in Eq. (11) of the main manuscript. The results for the Beauty, Cellphones, and Grocery datasets are shown in Figures 4, 5, and 6, respectively. For clarity and focus, we present results on three representative domains that reflect trends consistent with those observed across the full range of evaluated datasets.

We first vary the number of quantization levels \( L \) from 2 to 8 and report NDCG@10 in Figures 4a,5a, and 6a. Performance improves as \( L \) increases from 2 to 4, likely because longer sequences provide better capacity to capture fine-grained semantic information. However, further increasing \( L \) beyond 4 leads to performance degradation, which can be attributed to error accumulation in longer autoregressive sequences during recommendations.

Next, we evaluate codebook sizes \( T \in  \{ {64},{128},{256},{512}\} \) , with the results shown in Figures 4b,4b,6b. Performance generally improves with larger codebooks, as they provide greater flexibility and token diversity distinguishing items. However, excessively enlarging the codebooks may lead to performance degradation. This may be due to the increased sensitivity to noise in item semantics, which can cause the model to overfit to spurious or less meaningful patterns.

![16_235_155_1344_340_0.jpg](images/16_235_155_1344_340_0.jpg)

Figure 4: Sensitivity analysis on Beauty.

![16_235_583_1343_342_0.jpg](images/16_235_583_1343_342_0.jpg)

Figure 5: Sensitivity analysis on Cellphones.

We then examine the impact of the hyperparameter \( {\lambda }_{\mathrm{{RQ}}} \) in Eq.(11) of the main manuscript in terms of NDCG@10.As shown in Figure \( 4\mathrm{c},4\mathrm{c},6\mathrm{c} \) , performance peaks at \( {\lambda }_{\mathrm{{RQ}}} = 1 \) across datasets. The results suggest that setting \( {\lambda }_{\mathrm{{RQ}}} \) too high degrades performance, as it overemphasizes the residual quantizations that may be less relevant. Conversely, overly small \( {\lambda }_{\mathrm{{RO}}} \) may underutilize codebook supervision. Notably, these findings highlight the importance of carefully tuning \( {\lambda }_{\mathrm{{RQ}}} \) to balance codebook-based identifiers and optimize performance.

Finally, we analyze the effect of the hyperparameter \( {\lambda }_{\mathrm{{MI}}} \) in Eq.(11) of the main manuscript in terms of NDCG@10.As shown in Figures \( 4\mathrm{\;d},5\mathrm{\;d} \) , and \( 6\mathrm{\;d} \) , the highest NDCG@10 is achieved at \( {\lambda }_{\mathrm{{MI}}} = {0.03} \) . The results indicate that setting \( {\lambda }_{\mathrm{{MI}}} \) too high may degrade performance by overemphasizing mutual information calibration, potentially amplifying irrelevant variations. On the other hand, setting \( {\lambda }_{\mathrm{{MI}}} \) too low weakens the influence of mutual information preservation, limiting the semantic alignment of token representations. These findings highlight the importance of properly tuning \( {\lambda }_{\mathrm{{MI}}} \) to balance semantic preservation and generalization for optimal performance.

## 8. Empirical Validation of Theoretical Claims

We empirically validate the technical correctness and practical relevance of Theorems 1-3. Entropy analysis (Theorem 1) shows improved token entropy; quantization error comparison (Theorem 2) confirms a lower reconstruction error; and MI variance analysis (Theorem 3) demonstrates enhanced performance stability across domains.

Entropy analysis of token space (Supporting Theorem 1). We compare the entropy of the token distributions produced by UniTok and codebook-based methods. Specifically, we analyze how much entropy gain is contributed by the router module in UniTok based on Eq. (12) of the main manuscript. The entropy is calculated based on the frequency of full token combinations across all items.

<table><tr><td>Method</td><td>Token Space Entropy</td></tr><tr><td>Codebook-based methods</td><td>9.63</td></tr><tr><td>UniTok without the router</td><td>9.63</td></tr><tr><td>UniTok (full)</td><td>10.42</td></tr></table>

Table 9: Token space entropy comparison.

As shown in Table 9, we observe that the router alone contributes an additional 0.79 Hartleys of entropy (measured using base-10 logarithm), expanding the capacity of the token space compared to standard codebook-based methods.

![17_234_154_1345_342_0.jpg](images/17_234_154_1345_342_0.jpg)

Figure 6: Sensitivity analysis on Grocery.

![17_604_587_590_330_0.jpg](images/17_604_587_590_330_0.jpg)

Figure 7: Comparison of residual quantization loss over training epochs between UniTok and LETTER.

Quantization error comparison (Supporting Theorem 2). To empirically support Theorem 2, which states that UniTok yields a lower expected quantization error than that of standard codebook-based methods, we compare their quantization losses on multi-domain setting. As shown in Figure 7, UniTok consistently achieves a lower quantization error than the baseline case with a single set of codebooks. This confirms the theoretical insight that the TokenMoE architecture-by leveraging expert specialization-reduces a representation error and provides more precise item encoding. Notably, the expert-wise partitioning allows UniTok to partially compensate for domain-specific tokenization inaccuracies, leading to improved modeling fidelity in heterogeneous environments.

MI variance and performance stability (Supporting Theorem 3). To validate Theorem 3, which posits that the variability in downstream performance across domains is upper-bounded by the variance of MI, we empirically examine the relationship between the variance of MI and the maximum difference in task loss across domains. Specifically, we compute the variance of MI estimates \( {\widehat{I}}^{\left( k\right) } \) across domains, denoted as Var \( \left\lbrack  {\widehat{I}}^{\left( k\right) }\right\rbrack \) , and compare it against the observed performance variability \( \mathop{\max }\limits_{{i, j}}\left| {{\mathcal{L}}^{\left( i\right) } - {\mathcal{L}}^{\left( j\right) }}\right| \) (Refer to Eq. (14) of the main manuscript). As shown in Figure 8, we observe a strong positive correlation between \( \sqrt{\operatorname{Var}\left\lbrack  {\widehat{I}}^{\left( k\right) }\right\rbrack  } \) and performance variability, consistent with the Lipschitz continuity assumption. This trend indicates that lower MI variance leads to more stable performance across domains, supporting our theoretical claim. These results suggest that MI serves as a reliable indicator of multi-domain representation consistency and can guide the design of more robust multi-domain recommendation systems.

## References

Bao, K.; Zhang, J.; Wang, W.; Zhang, Y.; Yang, Z.; Luo, Y.; Chen, C.; Feng, F.; and Tian, Q. 2025. A Bi-step Grounding Paradigm for Large Language Models in Recommendation Systems. ACM Transactions on Recommender Systems, 3(4): 1-27.

Bao, K.; Zhang, J.; Zhang, Y.; Wang, W.; Feng, F.; and He, X. 2023. Tallrec: An effective and efficient tuning framework to align large language model with recommendation. In Proceedings of the 17th ACM Conference on Recommender Systems, 1007-1014.

Dai, S.; Shao, N.; Zhao, H.; Yu, W.; Si, Z.; Xu, C.; Sun, Z.; Zhang, X.; and Xu, J. 2023. Uncovering ChatGPT's Capabilities in Recommender Systems. In Proceedings of the 17th ACM Conference on Recommender Systems, 1126-1132.

Geng, S.; Liu, S.; Fu, Z.; Ge, Y.; and Zhang, Y. 2022. Recommendation as language processing (rlp): A unified pretrain, personalized prompt & predict paradigm (p5). In Proceedings of the 16th ACM conference on recommender systems, 299-315. Gong, Y.; Ding, X.; Su, Y.; Shen, K.; Liu, Z.; and Zhang, G. 2023. An unified search and recommendation foundation model for cold-start scenario. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management, 4595-4601.

![18_595_144_597_347_0.jpg](images/18_595_144_597_347_0.jpg)

Figure 8: Relationship between MI variance and performance gap.

Li, X.; Chen, C.; Zhao, X.; Zhang, Y.; and Xing, C. 2023a. E4srec: An elegant effective efficient extensible solution of large language models for sequential recommendation. arXiv preprint arXiv:2312.02443.

Li, Y.; Lin, X.; Wang, W.; Feng, F.; Pang, L.; Li, W.; Nie, L.; He, X.; and Chua, T.-S. 2024. A survey of generative search and recommendation in the era of large language models. arXiv preprint arXiv:2404.16924.

Li, Y.; Yang, N.; Wang, L.; Wei, F.; and Li, W. 2023b. Generative Retrieval for Conversational Question Answering. Information Processing & Management, 60(5): 103475.

Liao, J.; Li, S.; Yang, Z.; Wu, J.; Yuan, Y.; and Wang, X. 2023. LLaRA: Aligning Large Language Models with Sequential Recommenders. CoRR.

Lin, J.; Chen, B.; Wang, H.; Xi, Y.; Qu, Y.; Dai, X.; Zhang, K.; Tang, R.; Yu, Y.; and Zhang, W. 2024a. Clickprompt: CTR models are strong prompt generators for adapting language models to CTR prediction. In Proceedings of the ACM Web Conference 2024, 3319-3330.

Lin, J.; Shan, R.; Zhu, C.; Du, K.; Chen, B.; Quan, S.; Tang, R.; Yu, Y.; and Zhang, W. 2024b. Rella: Retrieval-enhanced large language models for lifelong sequential behavior comprehension in recommendation. In Proceedings of the ACM Web Conference 2024, 3497-3508.

Lin, X.; Wang, W.; Li, Y.; Feng, F.; Ng, S.-K.; and Chua, T.-S. 2024c. Bridging items and language: A transition paradigm for large language model-based recommendation. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 1816-1826.

Lin, X.; Wang, W.; Li, Y.; Yang, S.; Feng, F.; Wei, Y.; and Chua, T.-S. 2024d. Data-efficient Fine-tuning for LLM-based Recommendation. In Proceedings of the 47th international ACM SIGIR conference on research and development in information retrieval, 365-374.

Liu, Q.; Chen, N.; Sakai, T.; and Wu, X.-M. 2024. Once: Boosting content-based recommendation with both open-and closed-source large language models. In Proceedings of the 17th ACM International Conference on Web Search and Data Mining, 452-461.

Prakash, T.; Jalan, R.; Singh, B.; and Onoe, N. 2023. Cr-sorec: Bert driven consistency regularization for social recommendation. In Proceedings of the 17th ACM Conference on Recommender Systems, 883-889.

Qiu, Z.; Wu, X.; Gao, J.; and Fan, W. 2021. U-BERT: Pre-training user representations for improved recommendation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, 4320-4327.

Ren, X.; Wei, W.; Xia, L.; Su, L.; Cheng, S.; Wang, J.; Yin, D.; and Huang, C. 2024. Representation learning with large language models for recommendation. In Proceedings of the ACM Web Conference 2024, 3464-3475.

Wang, Y.; Ren, Z.; Sun, W.; Yang, J.; Liang, Z.; Chen, X.; Xie, R.; Yan, S.; Zhang, X.; Ren, P.; et al. 2024. Enhanced Generative Recommendation via Content and Collaboration Integration. arXiv -2v103.

Wei, W.; Ren, X.; Tang, J.; Wang, Q.; Su, L.; Cheng, S.; Wang, J.; Yin, D.; and Huang, C. 2024. LImrec: Large language models with graph augmentation for recommendation. In Proceedings of the 17th ACM International Conference on Web Search and Data Mining, 806-815.

Wu, C.; Wu, F.; Qi, T.; and Huang, Y. 2022. Userbert: Pre-training user model with contrastive self-supervision. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2087-2092.

Xi, Y.; Liu, W.; Lin, J.; Cai, X.; Zhu, H.; Zhu, J.; Chen, B.; Tang, R.; Zhang, W.; and Yu, Y. 2024. Towards open-world recommendation with knowledge augmentation from large language models. In Proceedings of the 18th ACM Conference on Recommender Systems, 12-22.

Zhai, J.; Mai, Z.; Wang, C.; Yang, F.; Zheng, X.; Li, H.; and Tian, Y. 2025. Multimodal Quantitative Language for Generative Recommendation. In ICLR, Singapore, April 24-28, 2025.

Zhang, J.; Xie, R.; Hou, Y.; Zhao, X.; Lin, L.; and Wen, J.-R. 2023. Recommendation as Instruction Following: A Large Language Model Empowered Recommendation Approach. ACM Transactions on Information Systems.

Zhang, Y.; Feng, F.; Zhang, J.; Bao, K.; Wang, Q.; and He, X. 2025. Collm: Integrating collaborative embeddings into large language models for recommendation. IEEE Transactions on Knowledge and Data Engineering.