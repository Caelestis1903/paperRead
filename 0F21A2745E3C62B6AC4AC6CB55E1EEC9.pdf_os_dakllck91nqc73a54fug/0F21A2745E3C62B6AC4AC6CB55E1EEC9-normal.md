# Personalized Multi-Interest Modeling for Cross-Domain Recommendation to Cold-Start Users

Xiaodong \( {\mathrm{{Li}}}^{1,2} \) , Jiawei Sheng \( {}^{1} \) , Jiangxia \( {\mathrm{{Cao}}}^{3} \) , Xinghua Zhang \( {}^{1,2} \) , Wenyuan Zhang \( {}^{1,2} \) , Yong Sun \( {}^{1,2} \) , Shirui \( {\mathrm{{Pan}}}^{4} \) , Zhihong \( {\mathrm{{Tian}}}^{5,6,7} \) and Tingwen Liu \( {}^{1,2 * } \)

\( {}^{1} \) Institute of Information Engineering, Chinese Academy of Sciences, Beijing, China \( {}^{2} \) School of Cyber Security, University of Chinese Academy of Sciences, Beijing, China \( {}^{3} \) Kuaishou Technology, Beijing, China

\( {}^{4} \) School of Information and Communication Technology, Griffith University, Queensland, Australia

\( {}^{5} \) Cyberspace Institute of Advanced Technology, Guangzhou University, Guangdong, China

\( {}^{6} \) Guangdong Key Laboratory of Industrial Control System Security, Guangdong, China

\( {}^{7} \) Huangpu Research School of Guangzhou University, Guangdong, China

\{lixiaodong, shengjiawei, zhangxinghua, zhangwenyuan, sunyong, liutingwen\}@iie.ac.cn,

jiangxiacao@gmail.com, s.pan@griffith.edu.au, tianzhihong@gzhu.edu.cn

Abstract-Cross-domain recommendation (CDR) has demonstrated to be an effective solution for alleviating the user cold-start issue. By leveraging rich user-item interactions available in a richly informative source domain, CDR could improve the recommendation performance for cold-start users in the target domain. Previous CDR approaches mostly adhere the Embedding and Mapping (EMCDR) paradigm, which learns a user-shared mapping function to transfer users' preference from the source domain to the target domain, neglecting users' personalized preference. Recent CDR approaches further leverage the meta-learning paradigm, considering the CDR task for each user independently and learning user-specific mapping functions for each user. However, they mostly learn representations for each user individually, which ignores the common preference between different users, neglecting valuable information for CDR. In addition, all these approaches usually summarize the user's preference into an overall representation, which can hardly capture the user's multi-interest preference. To this end, we propose a personalized multi-interest modeling framework for CDR to cold-start users, termed as NF-NPCDR. Specifically, we propose a personalized preference encoder that enhances the neural process (NP) with the normalizing flow (NF) to convert the Gaussian (unimodal) distribution to a multimodal distribution, providing a novel way to capture the user's personalized multi-interest preference. Then, we propose a common preference encoder with a preference pool to capture the common preference between different users. Furthermore, we introduce a stochastic adaptive decoder to incorporate both the personalized and common preference for cold-start users, adaptively modulating both preference for better recommendation. Experimental evaluations demonstrate that NF-NPCDR outperforms previous SOTA approaches in five benchmark CDR scenarios.

Index Terms-Cross-Domain Recommendation, Cold-Start Recommendation, Neural Process, Normalizing Flow.

![0_933_808_706_415_0.jpg](images/0_933_808_706_415_0.jpg)

Fig. 1. (A) An illustration of the common preference between different users. (B) An illustration of the NP principle and NF-NP principle.

## I. INTRODUCTION

Recommendation systems (RS) are extensively applied across various real-world web applications, such as Amazon (e-commerce), Kuaishou (online video) and Twitter (social media). Collaborative filtering (CF) stands out as an effective approach and has been implemented to RS, including matrix factorization approaches [1, 2] and modern deep neural network approaches [3, 4]. Nevertheless, CF-based methods typically encounter the persistent user cold-start issue [5], which makes it difficult to make recommendation for new emerging (cold-start) users with few user-item interactions. Recently, cross-domain recommendation (CDR) has been introduced as an effective solution to the user cold-start issue. Specifically, CDR enhances recommendations for cold-start users in the target domain by exploiting the extensive user-item interactions available in a richly informative source domain.

A mainstream strategy of CDR methods involves training the model with users overlapped between both domains, and subsequently making recommendations for cold-start users in the target domain. Recent CDR methods \( \lbrack 6,7,8,9,{10},{11},{12} \) , 13] mostly adhere the Embedding and Mapping (EMCDR [7]) paradigm, which first generates user/item embeddings from both domains respectively, and then learns a mapping function to transfer users' preference from the source domain to the target domain. However, these EMCDR-based methods usually learn a user-shared mapping function for all users, which neglects users' personalized preference, thus limiting the transfer effectiveness for CDR.

---

* Corresponding author.

This work was funded by the Youth Innovation Promotion Association of CAS (No.2021153), the National Natural Science Foundation of China (No.62406319), the Postdoctoral Fellowship Program of CPSF (No.GZC20232968), the Guangdong S&T Program under Grant 2024B0101010002, the National Natural Science Foundation of China (No.U2436208, No.62372129), and the Project of Guangdong Key Laboratory of Industrial Control System Security (2024B1212020010).

---

To capture the personalized preference of users, some meta-learning-based CDR methods [10, 14, 15] have been proposed, which consider the CDR task for each user independently, and thus learn user-specific mapping functions for each user by meta-learning paradigm. Typically, PTUPCDR [14] proposes a meta-network to generate the personalized mapping functions based on users representations to transfer the personalized preference for each user. However, these meta-learning-based methods learn representations for each user individually, which ignores the common preference between different users. In contrast, we recognize that the users can have common preference, which provides crucial information for CDR. For example in Fig. 1(A), both user \( a \) and \( b \) read fantasy books in the Book (source) domain, reflecting a common preference namely fantasy-interest. Besides, we can also observe that both users watch fantasy movies in the Movie (target) domain, exhibiting the same preference in the target domain. Therefore, learning the common preference between different users could acquire abundant information from their related users, thus benefit user's preference modeling.

Although the above methods have achieved excellent results, these EMCDR-based and meta-learning-based methods typically summarize user's preference as an overall representation, which can hardly reflect the user's multi-interest preference \( \left\lbrack  {{16},{17}}\right\rbrack \) . As shown in Fig. 1(B), user \( c \) has multiple interests in the Book (source) domain, namely fantasy-interest, romantic-interest and adventure-interest. Directly learning the user preference with a simple and deterministic representation may bias part of the user's interests and lead to unitary recommendation in target domain.

To this end, we seek the unification of neural process (NP) [18] and normalizing flow (NF) [19, 20] as a promising solution. In general, NP provides a neural-based approximation of stochastic processes, which can model uncertain distributions over prediction functions. As shown in Fig. 1(B), NP maps user's preference to a Gaussian distribution, and extends the meta-learning paradigm to transfer the user's personalized preference from the source to target domain. Besides, we further enhance neural process with the normalizing flow, which can convert the Gaussian (unimodal) distribution to a multimodal distribution, thus helping to capture the user's personalized multi-interest preference. In this way, the unification of NP and NF allows for learning more powerful CDR mapping functions.

Following the above idea, we propose a novel personalized multi-interest modeling framework for CDR to cold-start users, termed as NF-NPCDR. Specifically, we propose a personalized preference encoder that enhances the neural process with the normalizing flow to capture the user's personalized multi-interest preference. Then, we propose a common preference encoder with a preference pool to capture the common preference between different users. Furthermore, we devise a stochastic adaptive decoder to incorporate both the personalized and common preference for cold-start users, adaptively modulating both preference for better recommendation. To summarize the main contributions of this paper:

- We introduce a multi-interest modeling framework to address the cold-start issue in CDR with the unification of neural process and normalizing flow.

- We propose a personalized preference encoder to capture the user's personalized multi-interest preference. Then we propose a common preference encoder to capture the common preference between different users. A stochastic adaptive decoder is also designed to further incorporate both the personalized and common preference for cold-start users.

- Extensive experiments demonstrate the effectiveness and superiority of our NF-NPCDR in five CDR scenarios. Further experiments demonstrate the significant ability of NF-NPCDR to capture user's multi-interest preference and model the common preference between different users.

The rest of this paper is organized as follows. A summary of related works is shown in Section II. We briefly review the neural process and normalizing flow techniques, as well as the problem definition in Section III. Section IV shows the personalized multi-interest modeling framework of our model. We present the experimental results and analyses of our NF-NPCDR in Section V. We finally conclude our approach and give the future work in Section VI.

## II. RELATED WORK

Cross-domain recommendation \( \left\lbrack  {{21},{22},{23},{24}}\right\rbrack \) aims to improve the performance of recommendation to users in the target domain by utilizing rich auxiliary information from the source domain. Recent studies of CDR can generally be categorized into two distinct types based on the aiming user groups: one focuses on addressing the data sparsity issue \( \left\lbrack  {{25},2,{26},{27}}\right\rbrack \) for overlapping users, and the other concentrates on solving the cold-start issue \( \left\lbrack  {7,6,{12},{15},{28}}\right\rbrack \) for cold-start (i.e., non-overlapping) users. In this article, we focus on the cold-start issue in the CDR task.

To alleviate the cold-start issue, several CDR methods [7, 11, 6, 12, 13] following Embedding and Mapping paradigm have been proposed. Specifically, EMCDR [7] learns a mapping function between the source and target domains to transfer users' preference. SSCDR [6] proposes a semi-supervised mapping function to learn the cross-domain relationship. DCDCSR [11] employs the MF models and DNN to map the user and item latent factors across domains or systems. DCDIR [12] and HCDIR [13] construct heterogeneous information networks and leverage the user-item interactions to learn the mapping function. Recent several meta-learning-based methods [10, 14] also follow the Embedding and Mapping paradigm. For example, TMCDR [10] and PTUPCDR [14] propose a meta network to transfer the personalized preference of users. More recently, CDRIB [29] attempts to learn disentanglement representations via information bottleneck. REMIT [30] leverages user's multiple interests by multiple deterministic interest embeddings, but neglects the uncertainty modelling of user interests. Our NF-NPCDR belongs to the meta-learning-based method and has significant design differences with the above methods that we capture the user's personalized multi-interest preference and the common preference between different users.

Neural Process [18] combines the strengths of Gaussian process and neural networks to define distributions over functions and can estimate the uncertainty in their predictions. An increasing amount of researches \( \left\lbrack  {{31},{18},{32}}\right\rbrack \) have been proposed to enhance the expressiveness of the NP models. CNP [31] make predictions with only few training data points, and can be extended to large datasets. NP [18] is proposed to learn the latent random variable and can estimate the uncertainty of predictions. ANP [32] incorporates attention into NP to address the underfitting issue. The characteristics of NP have been applied on multi-task learning [33] and image recognition [34]. Recently, NP has been applied to the field of recommendation systems [35, 5, 36, 37, 38]. IDNP [35] leverages dilated convolutions and neural process to model user's short-term and long-term interests. CDRNP [38] leverages attention mechanism to bridge the source and target domains, and the neural process to capture the preference correlations among users. TANP [5] follows the meta-learning paradigm to make predictions for new tasks with neural process. INP [37] focuses on capturing user's diverse intentions from intrinsic-level with neural stochastic process. Nevertheless, NF-NPCDR directly transfers users' personalized preference from source domain to target domain via neural process, having a fundamentally different framework from the above methods.

Normalizing flows [19, 20] enable to convert a Gaussian (unimodal) distribution to a multimodal distribution by applying a sequence of invertible and differentiable mappings. There have been numerous works [39] applying normalizing flows in the field of statistics and machine learning. Planar and Radial flows [40] are used to approximate posterior distributions, which are favored for their simplicity and computational ease. These two flows are only suitable for low-dimensional situations. In order to deal with high-dimensional and highly structured data, RealNVP [41] is a suitable choice, which uses real-valued non-volume preserving transformations. MAF [42] generalizes RealNVP by building a sequence of autoregressive models, resulting in a type of normalizing flow well-suited for density estimation. Normalizing flows have been applied to many tasks, such as reinforcement learning [43, 44], graph generation [45] and graph completion [46]. In this paper, we further enhance neural process with the normalizing flow to model the user's personalized multi-interest preference for CDR to cold-start users.

## III. PRELIMINARY AND PROBLEM DEFINITION

This section first introduces the background knowledge of the neural process and normalizing flow, then formally defines our CDR problem settings.

## A. Preliminary

1) Neural Process: Neural Process (NP) [18, 31] represents a class of neural latent variable models that approximate stochastic processes \( f : X \rightarrow  Y \) using neural networks with robust parameterization capabilities. Consider a set of dataset \( \mathcal{T} = {\left\{  {x}_{i},{y}_{i}\right\}  }_{i = 1}^{\left| \mathcal{T}\right| } \) , the corresponding probability distribution over every \( \left( {{x}_{i},{y}_{i}}\right) \) can be defined as follows:

\[
{\rho }_{{x}_{1 : \left| \mathcal{T}\right| }}\left( {{y}_{1 : \left| \mathcal{T}\right| } = \int p\left( {{y}_{1 : \left| \mathcal{T}\right| } \mid  {x}_{1 : \left| \mathcal{T}\right| }, f}\right) p\left( f\right) {df}}\right) \tag{1}
\]

NP leverages neural networks (NNs) with a latent random variable \( z \) to parameterize the stochastic processes \( f \) . Specifically, NP splits the dataset \( \mathcal{T} \) into a support set \( \mathcal{C} = {\left\{  {x}_{i},{y}_{i}\right\}  }_{i = 1}^{\left| \mathcal{C}\right| } \) and a query set \( \mathcal{Q} = {\left\{  {x}_{i},{y}_{i}\right\}  }_{i = 1}^{\left| \mathcal{Q}\right| } \) . Given the support set \( \mathcal{C} \) , NP aims to make predictions on the query set \( \mathcal{Q} \) with \( p\left( {\mathbf{z} \mid  \mathcal{C}}\right) \) derived from \( \mathcal{C} \) . Consequently, we rewrite the Bayesian inference as follows:

\[
p\left( {{y}_{1 : \left| \mathcal{Q}\right| } \mid  {x}_{1 : \left| \mathcal{Q}\right| },\mathcal{C}}\right)  = \int p\left( {{y}_{1 : \left| \mathcal{Q}\right| } \mid  {x}_{1 : \left| \mathcal{Q}\right| },\mathbf{z}}\right) p\left( {\mathbf{z} \mid  \mathcal{C}}\right) d\mathbf{z}, \tag{2}
\]

Here we assume \( p\left( {\mathbf{z} \mid  \mathcal{C}}\right)  \sim  \mathcal{N}\left( {\mathbf{0},\mathbf{I}}\right) \) for simplicity in paradigmatic variational frameworks [47]. However, the true posterior distribution is intractable when optimising the model parameters. Thus, NP adopts amortized variational inference [47, 48] to derive the corresponding evidence lower-bound (ELBO) function as follows:

\[
\log p\left( {{y}_{1 : \left| \mathcal{Q}\right| } \mid  {x}_{1 : \left| \mathcal{Q}\right| },\mathcal{C}}\right)  = \log \int p\left( {\mathbf{z},{y}_{1 : \left| \mathcal{Q}\right| } \mid  {x}_{1 : \left| \mathcal{Q}\right| },\mathcal{C}}\right) d\mathbf{z}
\]

\[
\geq  {E}_{q\left( {\mathbf{z} \mid  \mathcal{C},\mathcal{Q}}\right) }\left\lbrack  {\log \frac{p\left( {\mathbf{z},{y}_{1 : \left| \mathcal{Q}\right| } \mid  {x}_{1 : \left| \mathcal{Q}\right| },\mathcal{C}}\right) }{q\left( {\mathbf{z} \mid  \mathcal{C},\mathcal{Q}}\right) }}\right\rbrack
\]

\[
= {E}_{q\left( {\mathbf{z} \mid  \mathcal{C},\mathcal{Q}}\right) }\left\lbrack  {\mathop{\sum }\limits_{{i = 1}}^{\left| \mathcal{Q}\right| }\log p\left( {{y}_{i} \mid  {x}_{i},\mathbf{z}}\right)  + \log \frac{p\left( {\mathbf{z} \mid  \mathcal{C}}\right) }{q\left( {\mathbf{z} \mid  \mathcal{C},\mathcal{Q}}\right) }}\right\rbrack \tag{3}
\]

\[
= \underset{\text{ desired log-likelihood }}{\underbrace{{E}_{q\left( {\mathbf{z} \mid  \mathcal{C},\mathcal{Q}}\right) }\log p\left( {{y}_{1 : \left| \mathcal{Q}\right| } \mid  {x}_{1 : \left| \mathcal{Q}\right| },\mathbf{z}}\right) }} - \underset{\text{ KL divergence }}{\underbrace{\mathrm{{KL}}\left( {q\left( {\mathbf{z} \mid  \mathcal{C},\mathcal{Q}}\right) \left| \right| p\left( {\mathbf{z} \mid  \mathcal{C}}\right) }\right) }},
\]

The derivation results of Eq. (3) can be clearly understood in terms of two components: the first focuses on achieving the desired CDR task, while the second ensures that the support set \( \mathcal{C} \) and the query set \( \mathcal{Q} \) following the same stochastic process.

2) Normalizing Flows: Normalizing Flows (NF) [19, 49, 46, 20] are a series of invertible and differentiable mappings used to convert probability distributions. Specifically, they convert the Gaussian (unimodal) distribution to a multimodal distribution. In the framework of NF, \( {\mathbf{z}}_{0} \) is drawn from the original distribution (e.g., \( {q}_{0}\left( {{\mathbf{z}}_{0} \mid  \mathcal{C},\mathcal{Q}}\right) \) ). The transformation of the original distribution is governed by an invertible and differentiable mappings, denoted as \( {g}_{k} : {\mathbb{R}}^{d} \rightarrow  {\mathbb{R}}^{d} \) . By applying a sequence of \( {g}_{k} \) with \( K \) steps, the final random variable \( {\mathbf{z}}_{K} \) is obtained as follows:

\[
{\mathbf{z}}_{K} = {g}_{K} \circ  \ldots  \circ  {g}_{2} \circ  {g}_{1}\left( {\mathbf{z}}_{0}\right)
\]

\[
{q}_{K}\left( {{\mathbf{z}}_{K} \mid  \mathcal{C},\mathcal{Q}}\right)  = {q}_{0}\left( {{\mathbf{z}}_{0} \mid  \mathcal{C},\mathcal{Q}}\right) \mathop{\prod }\limits_{{k = 1}}^{K}{\left| \det \frac{\partial {g}_{k}}{\partial {\mathbf{z}}_{k - 1}}\right| }^{-1}, \tag{4}
\]

where \( {q}_{K}\left( {{\mathbf{z}}_{K} \mid  \mathcal{C},\mathcal{Q}}\right) \) is the distribution of the final random variable \( {\mathbf{z}}_{K} \) , and \( {\left| \det \frac{\partial {g}_{k}}{\partial {\mathbf{z}}_{k - 1}}\right| }^{-1} \) is the determinant of the Jacobian of \( {g}_{k} \) at random variable \( {\mathbf{z}}_{k - 1} \) .

![3_156_144_715_407_0.jpg](images/3_156_144_715_407_0.jpg)

Fig. 2. An illustration of our NF-NPCDR during the training phase and testing phase implementing the NF-NP principle.

3) Unification of Neural Process and Normalizing Flow: In this paper, we consider to further enhance the neural process with normalizing flow. By integrating Eq. (4) into Eq. (3), we refine the ELBO function as follows:

\[
{E}_{q\left( {\mathbf{z} \mid  \mathcal{C},\mathcal{Q}}\right) }\log p\left( {{y}_{1 : \left| \mathcal{Q}\right| } \mid  {x}_{1 : \left| \mathcal{Q}\right| },\mathbf{z}}\right)  - \operatorname{KL}\left( {q\left( {\mathbf{z} \mid  \mathcal{C},\mathcal{Q}}\right) \parallel p\left( {\mathbf{z} \mid  \mathcal{C}}\right) }\right)
\]

\[
\simeq  {E}_{{q}_{0}\left( {{\mathbf{z}}_{0} \mid  \mathcal{C},\mathcal{Q}}\right) }\left\lbrack  {\log p\left( {{y}_{1 : \left| \mathcal{Q}\right| } \mid  {x}_{1 : \left| \mathcal{Q}\right| },{\mathbf{z}}_{K}}\right)  - \log \frac{{q}_{K}\left( {{\mathbf{z}}_{K} \mid  \mathcal{C},\mathcal{Q}}\right) }{p\left( {{\mathbf{z}}_{K} \mid  \mathcal{C}}\right) }}\right\rbrack
\]

\[
= \underset{\text{ desired log-likelihood }}{\underbrace{{E}_{{q}_{0}\left( {{\mathbf{z}}_{0} \mid  \mathcal{C},\mathcal{Q}}\right) }\log p\left( {{y}_{1 : \left| \mathcal{Q}\right| } \mid  {x}_{1 : \left| \mathcal{Q}\right| },{\mathbf{z}}_{K}}\right) }}
\]

\[
- \underset{\text{ KL divergence }}{\underbrace{{E}_{{q}_{0}\left( {{\mathbf{z}}_{0} \mid  \mathcal{C},\mathcal{Q}}\right) }\left\lbrack  {\log {q}_{0}\left( {{\mathbf{z}}_{0} \mid  \mathcal{C},\mathcal{Q}}\right)  - \mathop{\sum }\limits_{{k = 1}}^{K}\left| {\det \frac{\partial {g}_{k}}{\partial {\mathbf{z}}_{k - 1}}}\right|  - \log p\left( {{\mathbf{z}}_{K} \mid  \mathcal{C}}\right) }\right\rbrack  }},
\]

(5)

In terms of implementation, we demonstrate the NF-NP paradigm in a practical standpoint. Fig. 2 displays NF-NPCDR employing the NF-NP principle during both the training phase and testing phase. Specifically, during the training phase, the normalizing flow is adopted to convert the Gaussian (unimodal) distribution \( q\left( {\mathbf{z} \mid  \mathcal{C},\mathcal{Q}}\right) \) to a multimodal distribution \( {q}_{K}\left( {{\mathbf{z}}_{K} \mid  \mathcal{C},\mathcal{Q}}\right) \) , and then we use \( {\mathcal{L}}_{KL} \) in Eq. (5) to assimilate \( {q}_{K}\left( {{\mathbf{z}}_{K} \mid  \mathcal{C},\mathcal{Q}}\right) \) and \( p\left( {\mathbf{z} \mid  \mathcal{C}}\right) \) . During the testing phase, the ground truth \( {y}_{1 : \left| \mathcal{Q}\right| } \) in query set \( \mathcal{Q} \) is not available. We first use normalizing flow to convert the \( p\left( {\mathbf{z} \mid  \mathcal{C}}\right) \) to \( {p}_{K}\left( {{\mathbf{z}}_{K} \mid  \mathcal{C}}\right) \) , then \( {\mathbf{z}}_{K} \) sampled from the \( {p}_{K}\left( {{\mathbf{z}}_{K} \mid  \mathcal{C}}\right) \) is substituted for \( {q}_{K}\left( {{\mathbf{z}}_{K} \mid  \mathcal{C},\mathcal{Q}}\right) \) , and thus together with \( {x}_{1 : \left| \mathcal{Q}\right| } \) in the query set, our NF-NPCDR make recommendations by \( p\left( {{y}_{1 : \left| \mathcal{Q}\right| } \mid  {x}_{1 : \left| \mathcal{Q}\right| },{\mathbf{z}}_{K}}\right) \) .

## B. Problem Definition

This study considers a general CDR scenario with the source domain and the target domain. Let \( {\mathcal{D}}^{s} = \left( {{\mathcal{U}}^{s},{\mathcal{V}}^{s},{\mathcal{Y}}^{s}}\right) \) and \( {\mathcal{D}}^{t} = \left( {{\mathcal{U}}^{t},{\mathcal{V}}^{t},{\mathcal{Y}}^{t}}\right) \) denote the interaction data from source domain and target domain, where \( \mathcal{U},\mathcal{V} \) and \( \mathcal{Y} \in \; \{ 0,1,2,3,4,5{\} }^{\left| \mathcal{U}\right|  \times  \left| \mathcal{V}\right| } \) represent the user set, item set and rating matrix. Here \( {y}_{i, j} \in  \mathcal{Y} \) is the rating score, which indicates the user \( {u}_{i} \in  \mathcal{U} \) preference for selecting a particular item \( {v}_{j} \in  \mathcal{V} \) . Specifically, the user sets \( {\mathcal{U}}^{s} \) and \( {\mathcal{U}}^{t} \) include an overlapping subset of users, referred to as \( {\mathcal{U}}^{o} = {\mathcal{U}}^{s} \cap  {\mathcal{U}}^{t} \) , then we leverage \( {\mathcal{U}}^{s \smallsetminus  o} = {\mathcal{U}}^{s} \smallsetminus  {\mathcal{U}}^{o} \) to represent the cold-start (i.e., non-overlapping) users that exist only in the source domain.

TABLE I NOTATIONS.

<table><tr><td>Notation</td><td>Description</td></tr><tr><td>\( {\mathcal{D}}^{s} \) and \( {\mathcal{D}}^{t} \)</td><td>Interaction data</td></tr><tr><td>\( {\mathcal{U}}^{s} \) and \( {\mathcal{U}}^{t} \)</td><td>User set</td></tr><tr><td>\( {\mathcal{V}}^{s} \) and \( {\mathcal{V}}^{t} \)</td><td>Item set</td></tr><tr><td>\( {\mathcal{Y}}^{s} \) and \( {\mathcal{Y}}^{t} \)</td><td>Rating matrix</td></tr><tr><td>\( {\mathcal{U}}^{s} \smallsetminus  {}^{o} \)</td><td>Cold-start (i.e., non-overlapping) user set</td></tr><tr><td>\( {\mathcal{U}}^{o} \)</td><td>Overlapping user set across domains</td></tr><tr><td>\( {\mathcal{T}}_{i},{\mathcal{C}}_{i} \) and \( {\mathcal{Q}}_{i} \)</td><td>Task, Support set and Query set for user \( {u}_{i} \)</td></tr><tr><td>\( {\Omega }^{tr} \) and \( {\Omega }^{te} \)</td><td>Training task set and Testing task set</td></tr><tr><td>\( \left| {\mathcal{C}}_{i}\right| \) and \( \left| {\mathcal{Q}}_{i}\right| \)</td><td>Number of interactions in \( {\mathcal{C}}_{i} \) and \( {\mathcal{Q}}_{i} \)</td></tr><tr><td>\( {y}_{i, j}^{s} \) and \( {y}_{i, j}^{t} \)</td><td>Actual rating score of user \( {u}_{i} \) to item \( {v}_{j}^{s} \) and \( {v}_{j}^{t} \)</td></tr><tr><td>\( {\widehat{y}}_{i, j}^{t} \)</td><td>Prediction rating score of user \( {u}_{i} \) to item \( {v}_{i}^{t} \)</td></tr><tr><td>\( {d}_{1},{d}_{2},{d}_{3} \)</td><td>Embedding dimension</td></tr><tr><td>\( {\mathbf{\mu }}_{i} \)</td><td>Mean of the Gaussian distribution</td></tr><tr><td>\( {\mathbf{\sigma }}_{i} \)</td><td>Variance of the Gaussian distribution</td></tr><tr><td>\( \mathbf{\epsilon } \)</td><td>Gaussian noise</td></tr><tr><td>\( {g}_{K} \)</td><td>Invertible transformation function</td></tr><tr><td>\( K \)</td><td>Normalizing flow step</td></tr><tr><td>\( \mathcal{P} \)</td><td>Preference pool</td></tr><tr><td>\( N \)</td><td>Number of the soft cluster centroids</td></tr><tr><td>\( \mathcal{M} \)</td><td>Soft assignments matrix</td></tr><tr><td>\( \mathcal{D} \)</td><td>Auxiliary distribution</td></tr><tr><td>\( {\mathbf{\eta }}_{i}^{l} \) and \( {\mathbf{\delta }}_{i}^{l} \)</td><td>Modulation parameters of stochastic adaptive decoder for the \( l \) -th layer</td></tr><tr><td>\( \lambda \)</td><td>Hyper-parameter to balance \( {\mathcal{L}}_{c} \)</td></tr><tr><td>\( {\mathcal{L}}_{c} \)</td><td>KL loss between soft assignments \( \mathcal{M} \) and auxiliary distribution \( \mathcal{D} \)</td></tr><tr><td>\( {\mathcal{L}}_{\mathrm{{KL}},\mathrm{i}} \)</td><td>KL loss rewritten with the normalizing flow</td></tr><tr><td>\( {\mathcal{L}}_{{rec}, i} \)</td><td>Desired log-likelihood for rating prediction</td></tr><tr><td>\( \mathcal{L} \)</td><td>Overall loss function</td></tr></table>

Consequently, given the user-item interactions from both the source and target domains, CDR aims to predict the rating scores for cold-start users in the target domain by exploiting the extensive user-item interactions in the source domain. Formally, we define our CDR problem with task \( {\mathcal{T}}_{i} \) , which aims to make personalized recommendation for user \( {u}_{i} \in  \mathcal{U} \) . The concept of a task \( {\mathcal{T}}_{i} \) includes user’s interactions from source domain as the support set \( {\mathcal{C}}_{i} \) and interactions from the target domain as the query set \( {\mathcal{Q}}_{i} \) , i.e., \( {\mathcal{T}}_{i} = {\mathcal{C}}_{i} \cup  {\mathcal{Q}}_{i} \) . Here we denote \( {\mathcal{C}}_{i} = {\left\{  {u}_{i},{v}_{j}^{s},{y}_{i, j}^{s}\right\}  }_{j = 1}^{\left| {\mathcal{C}}_{i}\right| } \) and \( {\mathcal{Q}}_{i} = {\left\{  {u}_{i},{v}_{j}^{t},{y}_{i, j}^{t}\right\}  }_{j = 1}^{\left| {\mathcal{Q}}_{i}\right| } \) .

For the above purpose, we train our model on the training tasks \( {\Omega }^{tr} \) with overlapping users \( {u}_{i}^{o} \in  {\mathcal{U}}^{o} \) , i.e., we use \( {\mathcal{C}}_{i} \) from source domain and \( {\mathcal{Q}}_{i} \) from target domain to learn the cross domain knowledge. In the testing phase, our model makes recommendation to cold-start users \( {u}_{i}^{s \smallsetminus  o} \in  {\mathcal{U}}^{s \smallsetminus  o} \) for new testing tasks \( {\Omega }^{te} \) with only \( {\mathcal{C}}_{i} \) available, i.e., we use \( {\mathcal{C}}_{i} \) from source domain and the cross domain knowledge learned from \( {\Omega }^{tr} \) to predict the rating score \( {y}_{i, j}^{t} \) for each candidate item \( {v}_{j}^{t} \in  {\mathcal{V}}^{t} \) in \( {\mathcal{Q}}_{i} \) from the target domain. The notations used in this paper are summarized in Table I.

## IV. APPROACH

In this section, we present our proposed approach NF-NPCDR, which consists of four components: 1) An embedding layer to generate initialized user/item representations as the inputs of NF-NPCDR; 2) A personalized preference encoder to capture the user's personalized multi-interest preference; 3) A common preference encoder to capture the common preference between different users; 4) A stochastic adaptive decoder to incorporate both the personalized preference and common preference for cold-start users, ultimately generating predicted rating scores for candidate items in the target domain. The framework of NF-NPCDR is illustrated in Fig. 3.

![4_156_152_1480_580_0.jpg](images/4_156_152_1480_580_0.jpg)

Fig. 3. Overview of NF-NPCDR, including personalized preference encoder, common preference encoder and stochastic adaptive decoder. Both of \( {\mathcal{C}}_{i} \) and \( {\mathcal{T}}_{i} \) are encoded by the neural process encoder to generate the \( p\left( {{\mathbf{z}}_{i} \mid  {\mathcal{C}}_{i}}\right) \) and \( {q}_{0}\left( {{\mathbf{z}}_{i,0} \mid  {\mathcal{C}}_{i},{\mathcal{Q}}_{i}}\right) \) . The \( {q}_{0}\left( {{\mathbf{z}}_{i,0} \mid  {\mathcal{C}}_{i},{\mathcal{Q}}_{i}}\right) \) is further converted by the normalizing flow encoder to generate the \( {q}_{K}\left( {{\mathbf{z}}_{i, K} \mid  {\mathcal{C}}_{i},{\mathcal{Q}}_{i}}\right) .{\mathcal{C}}_{i} \) is encoded by the common preference encoder to generate the final preference representation \( {\mathbf{h}}_{i} \) . Finally, \( {\mathbf{h}}_{i} \) is used as input to the FiLM and \( {\mathbf{z}}_{i, K} \) is concatenated with \( \left( {{\mathbf{u}}_{i}^{o},{\mathbf{v}}_{j}^{t}}\right) \) in \( {\mathcal{Q}}_{i} \) to predict \( {\dot{y}}_{i, j}^{t} \) via stochastic adaptive decoder.

## A. Embedding Layer

The embedding layer embeds the representations of users and items into a low dimensional space. Specifically, we initialize user embeddings as \( {\mathbf{u}}_{i} \in  {\mathbb{R}}^{{d}_{1}} \) , item embeddings as \( {\mathbf{v}}_{j}^{s} \in  {\mathbb{R}}^{{d}_{1}} \) in the source domain and \( {\mathbf{v}}_{j}^{t} \in  {\mathbb{R}}^{{d}_{1}} \) in the target domain respectively, where \( {d}_{1} \) is the embedding dimension.

## B. Personalized Preference Encoder

This section introduces two parts: 1) a neural process (NP) encoder to generate the variational approximations over \( {\mathcal{T}}_{i} \) and \( {\mathcal{C}}_{i};2 \) ) a normalizing flow (NF) encoder to convert the Gaussian (unimodal) distribution to a multimodal distribution to capture the user's personalized multi-interest preference.

1) Neural Process Encoder: NP regards each task \( {\mathcal{T}}_{i} \) decoupled from the same stochastic process \( f \) . Specifically, we approximate \( f \) using a random vector \( {\mathbf{z}}_{i} \) , which is generated by neural networks. Thus, the neural process encoder aims to generate the variational prior \( p\left( {{\mathbf{z}}_{i} \mid  {\mathcal{C}}_{i}}\right) \) conditioned on support set \( {\mathcal{C}}_{i} \) and variational posterior \( q\left( {{\mathbf{z}}_{i} \mid  {\mathcal{C}}_{i},{\mathcal{Q}}_{i}}\right) \) with \( {\mathcal{T}}_{i} \) .

Taking \( {\mathcal{C}}_{i} \) as an example, for interactions in \( {\mathcal{C}}_{i} \) , we incorporate the \( \left( {{\mathbf{u}}_{i},{\mathbf{v}}_{j}^{s},{y}_{i, j}^{s}}\right) \) to approximate \( f \) using neural networks to produce corresponding representations \( {\mathbf{r}}_{i, j} \in  {\mathbb{R}}^{\left| {\mathcal{C}}_{i}\right|  \times  {d}_{2}} \) , which can be formulated as:

\[
{\mathbf{r}}_{i, j} = \operatorname{MLP}\left( {\left\lbrack  {{\mathbf{u}}_{i}\parallel {\mathbf{v}}_{j}^{s}\parallel {y}_{i, j}^{s}}\right\rbrack  ;\phi }\right) ,\;\left( {{\mathbf{u}}_{i},{\mathbf{v}}_{j}^{s},{y}_{i, j}^{s}}\right)  \in  {\mathcal{C}}_{i} \tag{6}
\]

where we use \( \operatorname{MLP}\left( {\cdot ;\phi }\right) \) parameterized by \( \phi \) to fuse the features of \( {\mathcal{C}}_{i} \) . Besides, \( \left\lbrack  {\cdot \left| \right|  \cdot  }\right\rbrack \) denotes the concatenation operation.

Afterword, to obtain representation \( {\mathbf{r}}_{i} \) , we aggregate \( {\mathbf{r}}_{i, j} \) with an order-invariance operation as in NP studies [18, 50]. Therefore, taking \( {\mathcal{C}}_{i} \) as an example, to derive \( p\left( {{\mathbf{z}}_{i} \mid  {\mathcal{C}}_{i}}\right) \) , we generate \( {\mathbf{r}}_{i} \in  {\mathbb{R}}^{{d}_{2}} \) with a mean function for computational efficiency, the aggregation process is as follows:

\[
{\mathbf{r}}_{i} = \frac{1}{\left| {\mathcal{C}}_{i}\right| }\mathop{\sum }\limits_{{j = 1}}^{\left| {\mathcal{C}}_{i}\right| }{\mathbf{r}}_{i, j} \tag{7}
\]

To obtain the random vector \( {\mathbf{z}}_{i} \) , we employ Gaussian distribution to generate \( {\mathbf{z}}_{i} \) as in Variational inference studies [47]. Therefore, we samples \( {\mathbf{z}}_{i} \sim  \mathcal{N}\left( {{\mathbf{\mu }}_{i}\left( {\mathbf{r}}_{i}\right) ,{\mathbf{\sigma }}_{i}\left( {\mathbf{r}}_{i}\right) }\right) \) as follows:

\[
{\mathbf{r}}_{i}^{\prime } = \operatorname{ReLU}\left( {\operatorname{MLP}\left( {\mathbf{r}}_{i}\right) }\right)
\]

\[
{\mathbf{\mu }}_{i}\left( {\mathbf{r}}_{i}\right)  = \operatorname{MLP}\left( {\mathbf{r}}_{i}^{\prime }\right) , \tag{8}
\]

\[
{\mathbf{\sigma }}_{i}\left( {\mathbf{r}}_{i}\right)  = {0.1} + {0.9} * \operatorname{Sigmoid}\left( {\operatorname{MLP}\left( {\mathbf{r}}_{i}^{\prime }\right) }\right) ,
\]

Given that the random variable \( {\mathbf{z}}_{i} \) is intractable during the training phase, we further adopt the reparameterization trick [47] to sample \( {\mathbf{z}}_{i} \) from the probability distribution, which allows for smooth backpropagation of gradients:

\[
{\mathbf{z}}_{i} = {\mathbf{\mu }}_{i}\left( {\mathbf{r}}_{i}\right)  + \mathbf{\epsilon } \odot  {\mathbf{\sigma }}_{i}\left( {\mathbf{r}}_{i}\right) ,\;\mathbf{\epsilon } \sim  \mathcal{N}\left( {0,\mathbf{I}}\right) , \tag{9}
\]

where \( \odot \) refers to element-wise multiplications. In this way, we can sample \( {\mathbf{z}}_{i} \) from \( p\left( {{\mathbf{z}}_{i} \mid  {\mathcal{C}}_{i}}\right) \) and \( q\left( {{\mathbf{z}}_{i} \mid  {\mathcal{C}}_{i},{\mathcal{Q}}_{i}}\right) \) . Besides, formulating the users' interactions from both the source and target domains within a unified stochastic process enables the capture of users' personalized preference across domains.

2) Normalizing Flow Encoder: In general, the Gaussian distribution modeled by neural process encoder is a unimodal distribution. However, users typically exhibit multiple interests in real-world CDR scenarios. Towards this end, we are setting our sights on the normalizing flow. Taking \( {\mathcal{C}}_{i} \) as an example, the key idea of normalizing flow is to convert the Gaussian (unimodal) distribution \( {p}_{0}\left( {{\mathbf{z}}_{i,0} \mid  {\mathcal{C}}_{i}}\right) \) to a multimodal distribution \( {p}_{K}\left( {{\mathbf{z}}_{i, K} \mid  {\mathcal{C}}_{i}}\right) \) .

Specifically, \( {\mathbf{z}}_{i,0} \) sampled through Eq. (9) is converted by \( K \) steps of invertible and differentiable mappings \( {g}_{k} \) to a multimodal distribution \( {p}_{K}\left( {{\mathbf{z}}_{i, K} \mid  {\mathcal{C}}_{i}}\right) \) :

\[
{\mathbf{z}}_{i, K} = {g}_{K} \circ  \ldots  \circ  {g}_{2} \circ  {g}_{1}\left( {\mathbf{z}}_{i,0}\right) ,
\]

\[
{p}_{K}\left( {{\mathbf{z}}_{i, K} \mid  {\mathcal{C}}_{i}}\right)  = {p}_{0}\left( {{\mathbf{z}}_{i,0} \mid  {\mathcal{C}}_{i}}\right) \mathop{\prod }\limits_{{k = 1}}^{K}{\left| \det \frac{\partial {g}_{k}}{\partial {\mathbf{z}}_{i, k - 1}}\right| }^{-1}, \tag{10}
\]

where \( {\mathbf{z}}_{i, K} \) contains more meaningful users’ personalized multi-interest preference compared to \( {\mathbf{z}}_{i,0} \) . We provide a detailed analysis of the user's multi-interest preference captured by the normalizing flow in section V-F and V-G.

## C. Common Preference Encoder

This module consists of two parts: 1) a preference identity network to produce an elementary preference representation; 2) a preference pool to learn the common preference between different users in the support set \( {\mathcal{C}}_{i} \) .

1) Preference Identity Network: Given the support set \( {\mathcal{C}}_{i} \) from source domain, the preference identify network tries to generate a elementary preference representation \( {\mathbf{e}}_{i} \in  {\mathbb{R}}^{{d}_{3}} \) by fusing the interactions in \( {\mathcal{C}}_{i} \) , which can be formulated as:

\[
{\mathbf{e}}_{i, j} = \operatorname{MLP}\left( {\left\lbrack  {{\mathbf{u}}_{i}\parallel {\mathbf{v}}_{j}^{s}\parallel {y}_{i, j}^{s}}\right\rbrack  ;\theta }\right) ,\;\left( {{\mathbf{u}}_{i},{\mathbf{v}}_{j}^{s},{y}_{i, j}^{s}}\right)  \in  {\mathcal{C}}_{i}
\]

\[
{\mathbf{e}}_{i} = \frac{1}{\left| {\mathcal{C}}_{i}\right| }\mathop{\sum }\limits_{{j = 1}}^{\left| {\mathcal{C}}_{i}\right| }{\mathbf{e}}_{i, j} \tag{11}
\]

where we use \( \operatorname{MLP}\left( {\cdot ;\theta }\right) \) parameterized by \( \theta \) to fuse the features of \( {\mathcal{C}}_{i} \) and the elementary preference representation \( {\mathbf{e}}_{i} \) is generated with the same operation in Eq. (7) for convenience.

2) Preference Pool: The preference pool \( \mathcal{P} = \; \left\lbrack  {{\mathbf{a}}_{1},\ldots ,{\mathbf{a}}_{N}}\right\rbrack   \in  {\mathbb{R}}^{{d}_{3} \times  N} \) serves as an additional resource which contains \( N \) randomly initialized and trainable soft cluster centroids as \( {\mathbf{a}}_{n} \in  {\mathbb{R}}^{{d}_{3}} \) . As suggested by \( \left\lbrack  {{51},5}\right\rbrack \) , we introduce an unsupervised algorithm aims to capture the common preference between different users, which operates through a two-step process. The first step involves generating soft cluster assignments between the elementary preference representation \( {e}_{i} \) and the preference pool. The second step focuses on updating the preference pool by integrating from high-confidence assignments with the help of an auxiliary distribution \( \mathcal{D} \) .

The Student's t-distribution [52] is used to measure the similarity between \( {\mathbf{e}}_{i} \) and \( {\mathbf{a}}_{n} \) as follows:

\[
{\mathbf{c}}_{in} = \frac{{\left( 1 + {\begin{Vmatrix}{\mathbf{e}}_{i} - {\mathbf{a}}_{n}\end{Vmatrix}}^{2}/\alpha \right) }^{-\frac{\alpha  + 1}{2}}}{\mathop{\sum }\limits_{{n}^{\prime }}{\left( 1 + {\begin{Vmatrix}{\mathbf{e}}_{i} - {\mathbf{a}}_{{n}^{\prime }}\end{Vmatrix}}^{2}/\alpha \right) }^{-\frac{\alpha  + 1}{2}}}, \tag{12}
\]

where \( \alpha \) are the degrees of freedom of the Student’s t-distribution. \( {\mathbf{c}}_{in} \) can be understood as representing the probability of allocating \( {\mathbf{e}}_{i} \) to \( {\mathbf{a}}_{n} \) , indicative of a soft assignment. The final preference representation \( {\mathbf{h}}_{i} \in  {\mathbb{R}}^{{d}_{3}} \) is generated as:

\[
{\mathbf{h}}_{i} = {\mathbf{e}}_{i} + \mathcal{P}{\mathbf{c}}_{i}^{\top } \tag{13}
\]

where \( {\mathbf{c}}_{i} \in  {\mathbb{R}}^{N} \) is the similarity between \( {\mathbf{e}}_{i} \) and \( \left\lbrack  {{\mathbf{a}}_{1},\ldots ,{\mathbf{a}}_{N}}\right\rbrack \) . Actually, the interactions in \( {\mathcal{C}}_{i} \) represent the personal preference of user \( {\mathbf{u}}_{i} \) , and different users may have common preference which can be reflected by \( {\mathbf{c}}_{i} \) through interacting with the preference pool \( \mathcal{P} \) . Consequently, the common preference between different users are comprised into the final preference representation \( {\mathbf{h}}_{i} \) through Eq. (13).

The soft assignments of all users in \( {\Omega }^{tr} \) constitute \( \mathcal{M} = \; \left\lbrack  {{\mathbf{c}}_{1},\ldots ,{\mathbf{c}}_{\left| {\Omega }^{tr}\right| }}\right\rbrack   \in  {\mathbb{R}}^{N \times  \left| {\Omega }^{tr}\right| } \) . To update the preference pool and refine the clusters, we propose an auxiliary distribution \( \mathcal{D} \) . Specifically, we define our objective as a KL divergence loss \( {\mathcal{L}}_{c} \) between the soft assignments \( \mathcal{M} \) and the auxiliary distribution \( \mathcal{D} \) as follows:

\[
{\mathcal{D}}_{in} = \frac{{\left( {\mathcal{M}}_{in}\right) }^{2}/\mathop{\sum }\limits_{i}{\mathcal{M}}_{in}}{\mathop{\sum }\limits_{{n}^{\prime }}\left( {\mathcal{M}}_{i{n}^{\prime }}\right) /\mathop{\sum }\limits_{i}{\mathcal{M}}_{i{n}^{\prime }}} \tag{14}
\]

\[
{\mathcal{L}}_{c} = \operatorname{KL}\left( {\mathcal{D}\parallel \mathcal{M}}\right)  = \mathop{\sum }\limits_{i}\mathop{\sum }\limits_{n}{\mathcal{D}}_{in}\log \frac{{\mathcal{D}}_{in}}{{\mathcal{M}}_{in}},
\]

Implementing \( {\mathcal{L}}_{c} \) enhances the accuracy of predictions and prioritizes users assigned with high confidence. We provide a detailed analysis of the common preference between different users captured by NF-NPCDR in section V-H.

## D. Stochastic Adaptive Decoder

To predict the rating score \( {\widehat{y}}_{i, j}^{t} \) with interactions in \( {\mathcal{Q}}_{i} \) , we have obtained the user's personalized multi-interest preference \( {\mathbf{z}}_{i, K} \) and common preference \( {\mathbf{h}}_{i} \) between different users. To enhance the integration of \( {\mathbf{z}}_{i, K} \) and \( {\mathbf{h}}_{i} \) for prediction, we introduce a stochastic adaptive decoder built upon the FiLM [53, 54, 5], capable of incorporating both the personalized and common preference for cold-start users.

For implementation, our stochastic adaptive decoder estimates the conditional likelihood \( p\left( {{y}_{i, j}^{t} \mid  {\mathbf{u}}_{i},{\mathbf{v}}_{j}^{t},{\mathbf{z}}_{i, K}}\right) \) . Initially, we generate two modulation parameters \( {\mathbf{\eta }}_{i} \) and \( {\mathbf{\delta }}_{i} \) over the representation \( {\mathbf{h}}_{i} \) , following which the stochastic adaptive decoder is formulated as follows:

\[
{\mathbf{\eta }}_{i}^{l} = \tanh \left( {\operatorname{MLP}\left( {\mathbf{h}}_{i}\right) }\right)
\]

(15)

\[
{\mathbf{\delta }}_{i}^{l} = \tanh \left( {\operatorname{MLP}\left( {\mathbf{h}}_{i}\right) }\right) ,
\]

\[
{\mathbf{w}}^{0} = \left\lbrack  {{\mathbf{u}}_{i}\begin{Vmatrix}{\mathbf{v}}_{j}^{t}\end{Vmatrix}{\mathbf{z}}_{i, K}}\right\rbrack
\]

\[
{\mathbf{w}}^{l + 1} = \operatorname{ReLU}\left( {{\mathbf{\eta }}_{i}^{l} \odot  \left( {\operatorname{MLP}\left( {\mathbf{w}}^{l}\right) }\right)  + {\mathbf{\delta }}_{i}^{l}}\right) ,
\]

where \( {\mathbf{w}}^{l} \) denotes the input to the decoder, i.e., \( l \) -th layer. Specifically, \( {\mathbf{\eta }}_{i}^{l} \) modulates the weight of \( {\mathbf{w}}^{l} \) , while \( {\mathbf{\delta }}_{i}^{l} \) dominates the weight of \( {\mathbf{h}}_{i} \) . Consequently, the stochastic adaptive decoder can effectively modulate the personalized and common preference for cold-start users, ultimately predicting the rating score \( {\widehat{y}}_{i, j}^{t} \) for the candidate item \( {\mathbf{v}}_{j}^{t} \) .

## E. Loss Function

We define the loss function of our NF-NPCDR with three components: the log-likelihood to accomplish the desired CDR task, the KL divergence refined with the normalizing flow to impose regularization and KL divergence between soft assignments \( \mathcal{M} \) and auxiliary distribution \( \mathcal{D} \) .

Algorithm 1 for NF-NPCDR in the training phase

---

Input: Training overlapping users set \( {\mathcal{U}}^{o} \) ; Items set \( {\mathcal{V}}^{s} \) and \( {\mathcal{V}}^{t} \) ; User-item

rating matrix \( {\mathcal{Y}}^{s} \) and \( {\mathcal{Y}}^{t} \) ; Training tasks set \( {\Omega }^{tr} \) .

Output: Model parameters \( \Theta \) .

		Initialize all model parameters.

		while not convergence do

			for \( {\mathcal{T}}_{i} \in  {\Omega }^{tr} \) do

				Construct \( {\mathcal{C}}_{i} \) and \( {\mathcal{Q}}_{i} \) from \( {\mathcal{T}}_{i} \) .

				Apply NP to generate \( {q}_{0}\left( {{\mathbf{z}}_{i,0} \mid  {\mathcal{C}}_{i},{\mathcal{Q}}_{i}}\right) \) in Eq. (8)-(9).

				Sample a \( {\mathbf{z}}_{i,0} \) from \( {q}_{0}\left( {{\mathbf{z}}_{i,0} \mid  {\mathcal{C}}_{i},{\mathcal{Q}}_{i}}\right) \) .

				Apply NF to generate \( {\mathbf{z}}_{i, K} \) in Eq. (10).

				Generate \( {\mathbf{h}}_{i} \) in Eq. (11)-(13).

				Predictions on \( {\mathcal{Q}}_{i} \) with \( {\mathbf{z}}_{i, K} \) and \( {\mathbf{h}}_{i} \) in Eq. (15).

				Generate \( p\left( {{\mathbf{z}}_{i} \mid  {\mathcal{C}}_{i}}\right) \) using \( {\mathcal{C}}_{i} \) in Eq. (8) and Eq. (9).

				Calculate \( {\mathcal{L}}_{\text{ rec, i }} \) in Eq. (16) and \( {\mathcal{L}}_{\mathrm{{KL}},\mathrm{i}} \) in Eq. (17).

			end for

			Calculate \( {\mathcal{L}}_{c} \) in Eq. (14).

			Use the overall loss \( \mathcal{L} \) in Eq. (18) to optimize \( \Theta \) .

		end while

---

The desired log-likelihood for rating prediction in Eq. (5) is usually measured by Mean Square Error (MSE) [5, 14] as:

\[
{\mathcal{L}}_{\mathrm{{rec}},\mathrm{i}} =  - {E}_{{q}_{0}\left( {{z}_{i,0} \mid  {\mathcal{C}}_{i},{\mathcal{Q}}_{i}}\right) }\log p\left( {{y}_{i,1 : \left| {\mathcal{Q}}_{i}\right| }^{t} \mid  {\mathbf{u}}_{i}^{o},{\mathbf{v}}_{1 : \left| {\mathcal{Q}}_{i}\right| }^{t},{z}_{i, K}}\right)
\]

\[
\propto  \frac{1}{\left| {\mathcal{Q}}_{i}\right| }\mathop{\sum }\limits_{{j = 1}}^{\left| {\mathcal{Q}}_{i}\right| }{\left( {y}_{i, j}^{t} - {\widehat{y}}_{i, j}^{t}\right) }^{2}, \tag{16}
\]

where we use overlapping users' interactions in target domain \( \left( {{\mathbf{u}}_{i}^{o},{\mathbf{v}}_{1 : \left| {\mathcal{Q}}_{i}\right| }^{t},{y}_{i,1 : \left| {\mathcal{Q}}_{i}\right| }^{t}}\right) \) to replace \( \left( {{x}_{1 : \left| \mathcal{Q}\right| },{y}_{1 : \left| \mathcal{Q}\right| }}\right) \) . Moreover, we minimize the KL divergence in Eq. (5) by introducing the regularization term \( {\mathcal{L}}_{\mathrm{{KL}},\mathrm{i}} \) , which constrains the approximate posterior of \( {\mathcal{Q}}_{i} \) toward the conditional prior of \( {\mathcal{C}}_{i} \) :

\[
{\mathcal{L}}_{\mathrm{{KL}},\mathrm{i}} = {E}_{{q}_{0}\left( {{\mathbf{z}}_{i,0} \mid  {\mathcal{C}}_{i},{\mathcal{Q}}_{i}}\right) }\left\lbrack  {\log {q}_{0}\left( {{\mathbf{z}}_{i,0} \mid  {\mathcal{C}}_{i},{\mathcal{Q}}_{i}}\right) }\right.
\]

\[
- \mathop{\sum }\limits_{{k = 1}}^{K}\left| {\det \frac{\partial {g}_{k}}{\partial {\mathbf{z}}_{i, k - 1}}}\right|  - \log p\left( {{\mathbf{z}}_{i, K} \mid  {\mathcal{C}}_{i}}\right) \rbrack \tag{17}
\]

In conclusion, the total optimization function is formulated as:

\[
\mathcal{L} = \frac{1}{\left| {\Omega }^{tr}\right| }\mathop{\sum }\limits_{{i = 1}}^{\left| {\Omega }^{tr}\right| }\left( {{\mathcal{L}}_{\mathrm{{rec}},\mathrm{i}} + {\mathcal{L}}_{\mathrm{{KL}},\mathrm{i}}}\right)  + \lambda {\mathcal{L}}_{c} \tag{18}
\]

where \( \lambda \) is a hyper-parameter (ranging from 0 to 1 ) to balance \( {\mathcal{L}}_{c} \) from Eq. (14). The training algorithm of NF-NPCDR is shown in Algorithm 1.

Note that in the testing phase, given a cold-start user \( {u}_{i}^{s \smallsetminus  o} \in  {\mathcal{U}}^{s \smallsetminus  o} \) , we first enhance the neural process with the normalizing flow to generate \( {p}_{K}\left( {{\mathbf{z}}_{i, K} \mid  {\mathcal{C}}_{i}}\right) \) from \( {\mathcal{C}}_{i} \) . Then, we sample a random variable \( {\mathbf{z}}_{i, K} \) from \( {p}_{K}\left( {{\mathbf{z}}_{i, K} \mid  {\mathcal{C}}_{i}}\right) \) . And we generate \( {\mathbf{h}}_{i} \) from the common preference encoder. Last, we take \( {\mathbf{z}}_{i, K},{\mathbf{h}}_{i} \) and \( \left( {{\mathbf{u}}_{i}^{s \smallsetminus  o},{\mathbf{v}}_{j}^{t}}\right) \) in \( {\mathcal{Q}}_{i} \) as inputs to the stochastic adaptive decoder to predict \( {\widehat{y}}_{i, j}^{t} \) for \( {\mathcal{Q}}_{i} \) . The testing algorithm of NF-NPCDR is shown in Algorithm 2.

## F. Complexity

In NF-NPCDR, all modules are parameterized by MLP, including the personalized preference encoder, common preference encoder and stochastic adaptive decoder. Therefore, our model keeps an efficient architecture for training and inference. Specifically, the complexity of each module can be approximated as \( \mathcal{O}\left( {l{d}^{3}}\right) \) , where \( l \) represents the number of MLP layers and \( d \) is the hidden size of each layer. The complexity of computing the Jacobian determinant of the normalizing flow in Eq. (10) is \( \mathcal{O}\left( {Kd}\right) \) , where \( K \) is the flow-steps. The calculation of final preference representation \( {\mathbf{h}}_{i} \) costs \( \mathcal{O}\left( {N{d}^{2}}\right) \) , where \( N \) represents the number of soft cluster centroids in preference pool. In addition, the FiLM requires only two weight metrics per layer in stochastic adaptive decoder, which is computationally efficient. Overall, NF-NPCDR is a lightweight and efficient framework.

Algorithm 2 for NF-NPCDR in the testing phase

---

Input: Testing cold-start users set \( {\mathcal{U}}^{s \smallsetminus  o} \) ; Items set \( {\mathcal{V}}^{s} \) and \( {\mathcal{V}}^{t} \) ; User-item

rating matrix \( {\mathcal{Y}}^{s} \) ; Testing tasks set \( {\Omega }^{te} \) ; Model parameters \( \Theta \) .

Output: Prediction rating score \( {\widehat{y}}_{i, j}^{t} \) .

	for \( {\mathcal{T}}_{i} \in  {\Omega }^{te} \) do

		Construct \( {\mathcal{C}}_{i} \) and \( {\mathcal{Q}}_{i} \) from \( {\mathcal{T}}_{i} \) .

			Apply NP to generate \( {p}_{0}\left( {{\mathbf{z}}_{i,0} \mid  {\mathcal{C}}_{i}}\right) \) in Eq. (8)-(9).

			Sample a \( {\mathbf{z}}_{i,0} \) from \( {p}_{0}\left( {{\mathbf{z}}_{i,0} \mid  {\mathcal{C}}_{i}}\right) \) .

			Apply NF to generate \( {\mathbf{z}}_{i, K} \) in Eq. (10).

		Generate \( {\mathbf{h}}_{i} \) in Eq. (11)-(13).

		Predictions on \( {\mathcal{Q}}_{i} \) with \( {\mathbf{z}}_{i, K} \) and \( {\mathbf{h}}_{i} \) in Eq. (15).

	end for

---

## V. EXPERIMENTS

In this section, we conduct extensive experiments on three real-world CDR scenarios to answer the following research questions (RQs): (1) RQ1: Compared with other state-of-the-art CDR models, does our model achieve the significant performance? (2) RQ2: Does the normalized flow proposed in our model really capture user's personalized multi-interest preference? (3) RQ3: Does the preference pool proposed in our model really capture the common preference between different users? (4) RQ4: Can directly leveraging the neural process to bridge the source and target domains really achieve better experimental performance? (5) RQ5: What is the effect of different normalizing flows on NF-NPCDR? Is there a trade-off between experimental performance and computational speed of normalizing flow? (6) RQ6: How does the computational cost of NF-NPCDR?

## A. Datasets and Metrics

Following most existing methods \( \lbrack 6,{15},{10},{14},{38}\rbrack \) , we evaluate the performance of our NF-NPCDR against other baselines on two real-world datasets (i.e., Amazon \( {}^{1} \) and Douban \( {}^{2} \) ) as follows:

- The Amazon dataset collects user review data from amazon.com, one of the largest e-commerce sites in the world. The collected data spans from May 1996 to July 2014.

- The Douban dataset is collected from douban.com, a popular service in China that provides ratings in various categories.

On the one hand, there have been many works [55, 17, \( {56},{57},{58},{59},{60}\rbrack \) using Amazon/Douban dataset to perform multi-interest recommendation. On the other hand, many works \( \left\lbrack  {{15},6,{38},7,{61},{14},{10}}\right\rbrack \) also adopt Amazon/Douban dataset to alleviate the cold-start issue. Therefore, the Amazon and Douban datasets are widely used in the field of recommendation systems to explore multi-interest cold-start case. In this paper, we choose Amazon and Douban datasets to verify the effectiveness of our model in capturing the multi-interest preference of cold-start users.

---

\( {}^{1} \) http://jmcauley.ucsd.edu/data/amazon/

\( {}^{2} \) https://www.douban.com/

---

TABLE II

STATISTICS OF FIVE CDR SCENARIOS ON AMAZON AND DOUBAN DATASETS (#OVERLAP DENOTES THE NUMBER OF OVERLAPPING USERS).

<table><tr><td>Scenarios</td><td>Domain</td><td>#Users</td><td>#Overlap</td><td>#Items</td><td>#Ratings</td></tr><tr><td rowspan="2">Scenario 1</td><td>\( {\mathcal{D}}^{s} \) : Movie</td><td>123,960</td><td rowspan="2">18,031</td><td>50,052</td><td>1,697,533</td></tr><tr><td>\( {\mathcal{D}}^{t} \) : Music</td><td>75,258</td><td>64,443</td><td>1,097,592</td></tr><tr><td rowspan="2">Scenario 2</td><td>\( {\mathcal{D}}^{s} \) : Book</td><td>603,668</td><td rowspan="2">37,388</td><td>367,982</td><td>8,898,041</td></tr><tr><td>\( {\mathcal{D}}^{t} \) : Movie</td><td>123,960</td><td>50,052</td><td>1,697,533</td></tr><tr><td rowspan="2">Scenario 3</td><td>\( {\mathcal{D}}^{s} \) : Book</td><td>603,668</td><td rowspan="2">16,738</td><td>367,982</td><td>8,898,041</td></tr><tr><td>\( {\mathcal{D}}^{t} \) : Music</td><td>75,258</td><td>64,443</td><td>1,097,592</td></tr><tr><td rowspan="2">Scenario 4</td><td>\( {\mathcal{D}}^{s} \) : Movie</td><td>2,712</td><td rowspan="2">2,209</td><td>34,893</td><td>1,278,401</td></tr><tr><td>\( {\mathcal{D}}^{t} \) : Book</td><td>2,212</td><td>95,872</td><td>227,251</td></tr><tr><td rowspan="2">Scenario 5</td><td>\( {\mathcal{D}}^{s} \) : Movie</td><td>2,712</td><td rowspan="2">1,815</td><td>34,893</td><td>1,278,401</td></tr><tr><td>\( {\mathcal{D}}^{t} \) : Music</td><td>1,820</td><td>79,878</td><td>17,9847</td></tr></table>

The Amazon dataset consists of 24 various item domains. Since users tend to have similar preferences in relevant domains, following previous methods [38, 30, 14], we select three relevant domains from Amazon, including Book (named "Books" in Amazon), Movie (named "Movie and TV" in Amazon) and Music (named "CDs and Vinyl" in Amazon) to form Scenario 1 : Amazon-Movie \( \rightarrow \) Amazon-Music, Scenario 2 : Amazon-Book \( \rightarrow \) Amazon-Movie, and Scenario 3 : Amazon-Book \( \rightarrow \) Amazon-Music. Similarly, we select three relevant domains from Douban, including Movie, Book and Music to form scenario 4: Douban-Movie \( \rightarrow \) Douban-Book and scenario 5: Douban-Movie \( \rightarrow \) Douban-Music. Both datasets contain rating scores from 1 to 5 , reflecting user's preference for specific items. For each CDR scenario, we first filter out users and items with fewer than 5 interactions. Then, we further filter out the items with user rating scores less than 4 in the source domain to eliminate noise information. Moreover, we create user interaction sequences in the source/target domain based on the sequential timestamps. Unlike many existing works \( \left\lbrack  {{62},3,{63},{64},1,6}\right\rbrack \) that only utilize a subset of the dataset for evaluation, we employ the entire dataset to better simulate real-world application scenarios. Table II summarizes the details of five CDR scenarios.

Following previous works \( \left\lbrack  {7,{61},{14}}\right\rbrack \) , We adopt two widely used metrics, Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) for performance comparison.

## B. Baselines

In our experiments, we compare our NF-NPCDR with the following state-of-the-art baselines: (1) TGT [65] denotes the target matrix factorization model, which concentrates solely on interactions within the target domain, ignoring those from the source domain. (2) CMF [2] is a collective matrix factorization method, involves decomposing multiple matrices simultaneously and sharing parameters among users across domains. (3) EMCDR [7] is the first work that proposes an Embedding and Mapping paradigm to cold-start users. It generates user/item representations in both domains, then utilizes a mapping function to align the users' representations. (4) CATN [61] proposes to facilitate the transfer of user preferences at the aspect level and employs an attention mechanism to learn the aspect correlations across domains. (5) DCDCSR [11] enhances the EMCDR paradigm by introducing benchmark factors. This method considers the varying degrees of rating sparsity for single users or items across different domains and systems. (6) SSCDR [6] introduces a CDR framework that utilizes a semi-supervised mapping approach. It models the cross-domain relationship with the help of mapping function. (7) LACDR [66] aims to reduce the training difficulty of the mapping function, which maintains high expressiveness and facilitates effective optimization. (8) RecGURU [67] proposes an adversarial learning method to achieve cross-domain collaboration in user representations. (9) PTUPCDR [14] uses a meta-learning-based framework to bridge the source and target domains with personalized mapping functions for each user. (10) REMIT [30] constructs a heterogeneous information network to generate user's multiple deterministic interest embeddings across domains. (11) CDRNP [38] uses the attention mechanism to bridge the source and target domains, then introduces the neural process to capture user preference correlations within domains.

In contrast, we exploit neural process to explicitly transfer users' personalized preference across domains, and further adopt normalizing flow to fully model the user's personalized multi-interest preference.

## C. Implementation Details

The experiments are conducted in a server with Intel(R) Xeon(R) Silver 4110 CPU @ 2.10GHz, and Tesla T4 (16G) GPU. The above methods are implemented in Pytorch 1.9.0 with python 3.6.8. In our experiments, we adjust the input to models and evaluation metrics of all baselines based on their official implementations to align with our experimental settings. To ensure the fairness of comparison, we adopt the identical value for the common hyper-parameters, including: the dimensions of the user/item representations are set to 10 , the learning rate is set to 0.01 , the batch size is set to 128 , the number of MLP layers is set to 3 and the hidden size of each fully connected layer is set to 64 . For the specific hyper-parameters in the baselines, we use the values reported in their original literature. For our NF-NPCDR, we select the Planar flow [40] to serve as the normalizing flow in Eq. (10). The number of the transformation steps \( K \) in Eq. (10) is chosen from \( \{ 2,4,6,8,{10},{12},{14},{16}\} \) . The number \( N \) of soft cluster centroids in the preference pool \( \mathcal{P} \) is chosen from \( \{ {10},{20},{30},{40},{50},{60}\} \) . The hyper-parameter \( \lambda \) in Eq. (18) is chosen from \( \{ {0.1},{0.2},{0.3},{0.4},{0.5},{0.6},{0.7},{0.8},{0.9},{1.0}\} \) , the length of the support set \( {\mathcal{C}}_{i} \) is chosen from \( \{ 5,{10},{15},{20},{25}\} \) . We consider sequential timestamps in our experiments to prevent information leakage. For each CDR scenario, the best hyper-parameters are tuned by grid search according to Mean Absolute Error (MAE) over five random runs.

TABLE III

OVERALL PERFORMANCE COMPARISON ON AMAZON DATASET.

<table><tr><td rowspan="3">Methods</td><td colspan="6">Amazon-Movie \( \rightarrow \) Amazon-Music</td><td colspan="6">Amazon-Book → Amazon-Movie</td><td colspan="6">Amazon-Book → Amazon-Music</td></tr><tr><td colspan="2">20%</td><td colspan="2">50%</td><td colspan="2">80%</td><td colspan="2">20%</td><td colspan="2">50%</td><td colspan="2">80%</td><td colspan="2">20%</td><td colspan="2">50%</td><td colspan="2">80%</td></tr><tr><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr><tr><td>TGT [65]</td><td>4.4803</td><td>5.1580</td><td>4.4989</td><td>5.1736</td><td>4.5020</td><td>5.1891</td><td>4.1831</td><td>4.7536</td><td>4.2288</td><td>4.7920</td><td>4.2123</td><td>4.8149</td><td>4.4873</td><td>5.1672</td><td>4.5073</td><td>5.1727</td><td>4.5204</td><td>5.2308</td></tr><tr><td>CMF [2]</td><td>1.5209</td><td>2.0158</td><td>1.6893</td><td>2.2271</td><td>2.4186</td><td>3.0936</td><td>1.3632</td><td>1.7918</td><td>1.5813</td><td>2.0886</td><td>2.1577</td><td>2.6777</td><td>1.8284</td><td>2.3829</td><td>2.1282</td><td>2.7275</td><td>3.0130</td><td>3.6948</td></tr><tr><td>DCDCSR [11]</td><td>1.4918</td><td>1.9210</td><td>1.8144</td><td>2.3439</td><td>2.7194</td><td>3.3065</td><td>1.3971</td><td>1.7346</td><td>1.6731</td><td>2.0551</td><td>2.3618</td><td>2.7702</td><td>1.8411</td><td>2.2955</td><td>2.1736</td><td>2.6771</td><td>3.1405</td><td>3.5842</td></tr><tr><td>SSCDR [6]</td><td>1.3017</td><td>1.6579</td><td>1.3762</td><td>1.7477</td><td>1.5046</td><td>1.9229</td><td>1.2390</td><td>1.6526</td><td>1.2137</td><td>1.5602</td><td>1.3172</td><td>1.7024</td><td>1.5414</td><td>1.9283</td><td>1.4739</td><td>1.8441</td><td>1.6414</td><td>2.1403</td></tr><tr><td>EMCDR [7]</td><td>1.2350</td><td>1.5515</td><td>1.3277</td><td>1.6644</td><td>1.5008</td><td>1.8771</td><td>1.1162</td><td>1.4120</td><td>1.1832</td><td>1.4981</td><td>1.3156</td><td>1.6433</td><td>1.3524</td><td>1.6737</td><td>1.4723</td><td>1.8000</td><td>1.7191</td><td>2.1119</td></tr><tr><td>CATN [61]</td><td>1.2671</td><td>1.6468</td><td>1.4890</td><td>1.9205</td><td>1.8182</td><td>2.2991</td><td>1.1249</td><td>1.4548</td><td>1.1598</td><td>1.4826</td><td>1.2672</td><td>1.6280</td><td>1.3924</td><td>1.7399</td><td>1.6023</td><td>2.0665</td><td>1.9571</td><td>2.5623</td></tr><tr><td>LACDR [66]</td><td>1.1295</td><td>1.4358</td><td>1.3502</td><td>1.7510</td><td>1.6886</td><td>2.2238</td><td>0.9681</td><td>1.2311</td><td>1.0077</td><td>1.3051</td><td>1.1151</td><td>1.4660</td><td>1.1945</td><td>1.5771</td><td>1.3925</td><td>1.8644</td><td>1.7107</td><td>2.2468</td></tr><tr><td>RecGURU [67]</td><td>1.2320</td><td>1.5545</td><td>1.3696</td><td>1.6640</td><td>1.7154</td><td>2.2160</td><td>1.0404</td><td>1.2598</td><td>1.2434</td><td>1.4921</td><td>1.2243</td><td>1.5801</td><td>1.4467</td><td>1.7188</td><td>1.7496</td><td>2.0766</td><td>1.8535</td><td>2.3401</td></tr><tr><td>PTUPCDR [14]</td><td>1.1504</td><td>1.5195</td><td>1.2804</td><td>1.6380</td><td>1.4049</td><td>1.8234</td><td>0.9970</td><td>1.3317</td><td>1.0894</td><td>1.4395</td><td>1.1999</td><td>1.5916</td><td>1.2286</td><td>1.6085</td><td>1.3764</td><td>1.7447</td><td>1.5784</td><td>2.0510</td></tr><tr><td>REMIT [30]</td><td>0.9393</td><td>1.2709</td><td>1.0437</td><td>1.4580</td><td>1.2181</td><td>1.6601</td><td>0.8759</td><td>1.1650</td><td>0.9172</td><td>1.2379</td><td>1.0055</td><td>1.3772</td><td>1.3749</td><td>1.9940</td><td>1.4401</td><td>2.0495</td><td>1.6396</td><td>2.2653</td></tr><tr><td>CDRNP [38]</td><td>0.7974</td><td>1.0638</td><td>0.7969</td><td>1.0589</td><td>0.8280</td><td>1.0758</td><td>0.8846</td><td>1.1327</td><td>0.8946</td><td>1.1450</td><td>0.8970</td><td>1.1576</td><td>0.7453</td><td>0.9914</td><td>0.7629</td><td>1.0111</td><td>0.7787</td><td>1.0373</td></tr><tr><td>NF-NPCDR</td><td>0.4348</td><td>0.8146</td><td>0.4479</td><td>0.8310</td><td>0.4598</td><td>0.8399</td><td>0.4490</td><td>0.8511</td><td>0.4445</td><td>0.8643</td><td>0.4746</td><td>0.8828</td><td>0.3894</td><td>0.7632</td><td>0.4027</td><td>0.7706</td><td>0.4170</td><td>0.7748</td></tr><tr><td>Improve</td><td>45.47%</td><td>23.43%</td><td>43.79%</td><td>21.52%</td><td>44.47%</td><td>21.93%</td><td>48.74%</td><td>24.86%</td><td>50.31%</td><td>24.52%</td><td>47.09%</td><td>23.74%</td><td>47.75%</td><td>23.02%</td><td>47.21%</td><td>23.79%</td><td>46.45%</td><td>25.31%</td></tr></table>

TABLE IV

OVERALL PERFORMANCE COMPARISON ON DOUBAN DATASET.

<table><tr><td rowspan="3">Methods</td><td colspan="6">Douban-Movie \( \rightarrow \) Douban-Book</td><td colspan="6">Douban-Movie \( \rightarrow \) Douban-Music</td></tr><tr><td colspan="2">20%</td><td colspan="2">50%</td><td colspan="2">80%</td><td colspan="2">20%</td><td colspan="2">50%</td><td colspan="2">80%</td></tr><tr><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr><tr><td>TGT [65]</td><td>4.3608</td><td>5.1976</td><td>4.2722</td><td>5.0742</td><td>4.2999</td><td>5.1121</td><td>4.3225</td><td>5.1000</td><td>4.3590</td><td>5.1262</td><td>4.3877</td><td>5.1655</td></tr><tr><td>CMF [2]</td><td>2.3455</td><td>3.1429</td><td>2.5414</td><td>3.3250</td><td>3.0902</td><td>3.8110</td><td>2.5785</td><td>3.3743</td><td>2.7830</td><td>3.5396</td><td>3.2192</td><td>3.9194</td></tr><tr><td>DCDCSR [11]</td><td>2.2801</td><td>2.7748</td><td>2.5499</td><td>3.0822</td><td>3.3885</td><td>3.6213</td><td>2.1690</td><td>2.5646</td><td>2.7990</td><td>3.1982</td><td>3.6585</td><td>3.8855</td></tr><tr><td>SSCDR [6]</td><td>2.2723</td><td>2.8064</td><td>2.4033</td><td>2.9383</td><td>3.2540</td><td>3.4689</td><td>2.2496</td><td>2.5602</td><td>3.0579</td><td>2.9516</td><td>3.3226</td><td>3.4409</td></tr><tr><td>EMCDR [7]</td><td>2.4435</td><td>3.0077</td><td>2.6365</td><td>3.0877</td><td>3.4904</td><td>3.7434</td><td>2.3045</td><td>2.8589</td><td>2.6926</td><td>3.1398</td><td>3.5168</td><td>3.7645</td></tr><tr><td>CATN [61]</td><td>2.3285</td><td>2.9363</td><td>2.4721</td><td>2.9764</td><td>3.2250</td><td>3.4573</td><td>2.2685</td><td>2.7831</td><td>2.8840</td><td>3.0276</td><td>3.3078</td><td>3.4930</td></tr><tr><td>LACDR [66]</td><td>2.2032</td><td>2.7317</td><td>2.8873</td><td>3.4064</td><td>2.9873</td><td>3.2064</td><td>2.8020</td><td>3.3021</td><td>3.0005</td><td>3.3544</td><td>3.4071</td><td>3.6752</td></tr><tr><td>RecGURU [67]</td><td>2.2579</td><td>2.8931</td><td>2.6720</td><td>3.1351</td><td>3.0704</td><td>3.2916</td><td>2.2943</td><td>2.6009</td><td>2.7163</td><td>2.9245</td><td>3.2880</td><td>3.4061</td></tr><tr><td>PTUPCDR [14]</td><td>1.6914</td><td>2.3176</td><td>1.9729</td><td>2.5599</td><td>2.8191</td><td>3.2046</td><td>1.9116</td><td>2.5670</td><td>2.1192</td><td>2.7271</td><td>2.8990</td><td>3.3057</td></tr><tr><td>REMIT [30]</td><td>1.4628</td><td>2.0757</td><td>1.8023</td><td>2.3849</td><td>2.7584</td><td>2.9835</td><td>1.6657</td><td>2.4962</td><td>2.0270</td><td>2.5538</td><td>2.7376</td><td>3.1890</td></tr><tr><td>CDRNP [38]</td><td>0.7350</td><td>0.9315</td><td>0.7363</td><td>0.9470</td><td>0.7382</td><td>0.9461</td><td>0.6971</td><td>0.8937</td><td>0.7376</td><td>0.9278</td><td>0.7505</td><td>0.9428</td></tr><tr><td>NF-NPCDR</td><td>0.4236</td><td>0.4861</td><td>0.4148</td><td>0.4703</td><td>0.4299</td><td>0.4839</td><td>0.4339</td><td>0.5012</td><td>0.4233</td><td>0.4813</td><td>0.4581</td><td>0.4975</td></tr><tr><td>Improve</td><td>42.37%</td><td>47.82%</td><td>43.66%</td><td>50.34%</td><td>41.76%</td><td>48.85%</td><td>37.76%</td><td>43.92%</td><td>42.61%</td><td>48.12%</td><td>38.96%</td><td>47.23%</td></tr></table>

Following previous methods [7, 14, 38], we randomly remove all the ratings from a subset of overlapping users in the target domain, treating these users as testing (cold-start) users, while considering the remaining overlapping users as training users. Suggested by previous methods [38, 30, 14], we set \( {20}\% ,{50}\% \) , and \( {80}\% \) of the overlapping users as the proportion of testing (cold-start) users, denoted as \( \alpha \) . In fact, different overlapping user ratio (i.e., \( 1 - \alpha \) ) represents different situations, e.g., \( \alpha  = {20}\% \) represents most of users are overlapped, while \( \alpha  = {80}\% \) means only few users are overlapped. Following previous meta-learning recommenders [68, 69, 70], we focus on predicting the ratings of each user for the candidate items in the query set \( {\mathcal{Q}}_{i} \) .

## D. Performance Comparisons (RQ1)

Table III and IV show the comparison results on five CDR scenarios according to MAE and RMSE. From the experimental results, we observe that NF-NPCDR shows consistent prediction improvement compared with the single-domain baselines (i.e., TGT). This observation indicates that devising information transfer strategies across domain is effective to improve the recommendation performance in target domain. We also observe that NF-NPCDR consistently outperforms the cross-domain baselines in all five CDR scenarios, indicating the effectiveness of our model. Specifically, with different \( \alpha \) value settings at \( {20}\% ,{50}\% \) and \( {80}\% \) , NF-NPCDR surpasses the current state-of-the-art method (i.e., CDRNP) in terms of MAE by an average of 47.32%, 47.10% and 46.00% in Amazon dataset, and 40.07%, 43.14% and 40.36% in Douban dataset, respectively. This observation validates that capturing the user's personalized multi-interest preference and common preference between different users are effective for CDR.

We perform a further introspection of our results and note that the multimodal distribution modeled by NF-NPCDR can capture richer and more meaningful information for CDR. Compared to the EMCDR-based CDR methods, our NF-NPCDR achieves better performance, which indicates that a multimodal distribution can capture the user's multi-interest preference. Meanwhile, we find that compared with the SOTA meta-learning method (i.e., CDRNP), NF-NPCDR exhibits exceptional performance and robustness, even in scenarios where the training data (overlapping users) is limited. Specifically, for Amazon-Movie \( \rightarrow \) Amazon-Music, even if the \( \alpha \) value of the SOTA meta-learning method is set to \( {20}\% \) and the \( \alpha \) value of our method is set to \( {80}\% \) , we can still obtain an experimental performance improvement in MAE of 42.34%. In other words, our method uses only 20% training data can significantly outperform the SOTA method with 80% training data. This indicates that our normalizing flow enhanced neural process framework is capable of adapting to the CDR task with only a limited number of overlapping users.

TABLE V

ABLATION STUDY ON FIVE CDR SCENARIOS.

<table><tr><td>Variants</td><td>Metric</td><td>Scenario1</td><td>Scenario2</td><td>Scenario3</td><td>Scenario4</td><td>Scenario5</td></tr><tr><td rowspan="2">NF-NPCDR <br> (NF+PO+FM)</td><td>MAE</td><td>0.4348</td><td>0.4490</td><td>0.3894</td><td>0.4236</td><td>0.4339</td></tr><tr><td>RMSE</td><td>0.8146</td><td>0.8511</td><td>0.7632</td><td>0.4861</td><td>0.5012</td></tr><tr><td rowspan="2">w/o NF</td><td>MAE</td><td>0.4660</td><td>0.4606</td><td>0.4132</td><td>0.4585</td><td>0.4671</td></tr><tr><td>RMSE</td><td>0.8320</td><td>0.8647</td><td>0.7761</td><td>0.5082</td><td>0.5209</td></tr><tr><td rowspan="2">w/o PO</td><td>MAE</td><td>0.4722</td><td>0.4689</td><td>0.4127</td><td>0.4510</td><td>0.4574</td></tr><tr><td>RMSE</td><td>0.8404</td><td>0.8605</td><td>0.7781</td><td>0.5074</td><td>0.5188</td></tr><tr><td rowspan="2">w/o FM</td><td>MAE</td><td>0.4468</td><td>0.4561</td><td>0.4009</td><td>0.4476</td><td>0.4513</td></tr><tr><td>RMSE</td><td>0.8224</td><td>0.8592</td><td>0.7697</td><td>0.4930</td><td>0.5119</td></tr><tr><td rowspan="2">W/o NF, PO, FM</td><td>MAE</td><td>0.4922</td><td>0.4865</td><td>0.4304</td><td>0.4770</td><td>0.4899</td></tr><tr><td>RMSE</td><td>0.8512</td><td>0.8735</td><td>0.7842</td><td>0.5215</td><td>0.5473</td></tr></table>

## E. Ablation Study

In this section, we conduct ablation studies to analyze the effectiveness of each component in NF-NPCDR. We conduct four variants of NF-NPCDR in Table V. Specifically, the \( w/o \) NF is the variant without normalizing flow encoder, the \( w/o \) PO is the variant without preference pool, the w/o FM is the variant without FiLM, and the w/o NF, PO, FM is the variant without the three components mentioned above. We set the \( \alpha \) value for all CDR scenarios to \( {20}\% \) .

Table V presents the results. We find that MAE of our model drops by \( {6.69}\% ,{2.52}\% ,{5.76}\% ,{7.61}\% \) and 7.11 \( \% \) after the normalizing flow encoder is removed in five CDR scenarios. This demonstrates that the importance of normalizing flow encoder to further capture the user's personalized multi-interest preference. We also observe that MAE of our model drops by \( {7.92}\% ,{4.24}\% ,{5.65}\% ,{6.08}\% \) and 5.14% after the preference pool is removed. This indicates that the common preference between different users captured by the preference pool is effective for CDR. Meanwhile, MAE of our model drops 2.69%, 1.56%, 2.87%, 5.36% and 3.86% after the FiLM is removed, which demonstrates that the FiLM can indeed incorporate both the personalized and common preference for cold-start users, adaptively modulate both preference for better recommendation. Lastly, compared to \( w/o\mathrm{{NF}},\mathrm{{PO}},\mathrm{{FM}} \) variant, the normalizing flow, preference pool and FiLM can contribute a total of 11.66%, 7.71%, 9.53%, 11.19% and 11.43% increase in MAE to our model. Thus, we should combine them to achieve the best performance and generalizability.

## F. Multimodal Distribution Analysis (RQ2)

In this section, we further investigate the ability of NF-NPCDR to model a multimodal distribution of CDR prediction functions, which captures the user's personalized multi-interest preference. The distribution of CDR prediction functions can be reflected by the entropy of \( {\mathbf{z}}_{i} \) . As found by Luo et al. [46], the multimodal distribution leads to a higher entropy than the Gaussian (unimodal) distribution. Therefore, the entropy has a positive correlation with the characteristics of the generated multimodal distribution. The higher the entropy, the richer the multimodal distribution. Specifically, we first evaluate the performance of NF-NPCDR and NF-NPCDR (w/o NF) with different lengths of support set \( {\mathcal{C}}_{i} \) in Fig. 4, then illustrate the corresponding entropy estimated by them in Fig. 5.

![9_937_137_705_223_0.jpg](images/9_937_137_705_223_0.jpg)

Fig. 4. Performance comparison of NF-NPCDR, NF-NPCDR (w/o NF) and CDRNP with different lengths of support set \( {\mathcal{C}}_{i} \) on Amazon dataset.

![9_934_450_709_221_0.jpg](images/9_934_450_709_221_0.jpg)

Fig. 5. Entropy \( \left( {\mathbf{z}}_{i}\right) \) estimated by NF-NPCDR and NF-NPCDR (w/o NF) with different lengths of support set \( {\mathcal{C}}_{i} \) on Amazon dataset.

From the results shown in Fig 4, we can see that NF-NPCDR consistently outperforms NF-NPCDR (w/o NF) with different lengths of support set \( {\mathcal{C}}_{i} \) . The reason is that normalizing flow can effectively model a multimodal distribution to capture the user's multi-interest preference compared to neural process. This phenomenon can also be illustrated in Fig. 5, the Entropy \( \left( {\mathbf{z}}_{i, K}\right) \) of NF-NPCDR is always higher than the Entropy \( \left( {\mathbf{z}}_{i,0}\right) \) of NF-NPCDR (w/o NF). This means that the multimodal distribution can capture richer and more meaningful preference information. Another observation is that both NF-NPCDR and NF-NPCDR (w/o NF) maintain stable performance when the interactions in \( {\mathcal{C}}_{i} \) is decreased. This indicates that our neural process-based meta-learning paradigm is less sensitive to the decrease of interactions in \( {\mathcal{C}}_{i} \) . This robustness is likely attributable to the use of neural process paradigm, which applies stochastic process to model a Gaussian distribution over user's preference, making our neural process-based meta-learning paradigm more stable.

## G. Case Study for Multi-Interest Preference (RQ2)

To better understand the user's personalized multi-interest preference captured by the normalizing flow, we conduct a case study. Specifically, we randomly select a cold-start user from the Amazon-Book (source domain) who has no interactions in the Amazon-Movie (target domain). The three interests preference of this cold-start user from the Book domain are Adventure (four books), Romantic (three books) and Fantasy (ten books), represented by orange, blue and red, respectively. And the ratings of this cold-start user to these books are all 5 , which are given in the datasets. Finally, the ratings of three candidate movies belonging to Adventure, Romantic and Fantasy are predicted by models.

TABLE VI

A CASE STUDY OF A COLD-START USER FOR CDR FROM AMAZON-BOOK TO AMAZON-MOVIE.

<table><tr><td>Amazon-Book <br> Amazon-Movie</td><td>Interest 1: Adventure, Number: 4 E.g., The Mystery of the Burnt Cottage</td><td>Interest 2: Romantic, Number: 3 E.g., Clouds Among the Stars</td><td>Interest 3: Fantasy, Number: 10 E.g., Dragon Mage</td></tr><tr><td>NF-NPCDR (Ours) NF-NPCDR w/o NF CDRNP PTUPCDR</td><td><img src="https://cdn.noedgeai.com/bo_daklld491nqc739pqp1g_10.jpg?x=606&y=302&w=70&h=73&r=0"/> 1.57 3.11 3.11 <br> 4.22.11 1.93</td><td><img src="https://cdn.noedgeai.com/bo_daklld491nqc739pqp1g_10.jpg?x=966&y=301&w=56&h=64&r=0"/> 4.32 2.61 <br> True Romance 2.12</td><td><img src="https://cdn.noedgeai.com/bo_daklld491nqc739pqp1g_10.jpg?x=1283&y=299&w=71&h=74&r=0"/> 3.86 <br> Rick and Morty 3.60</td></tr></table>

![10_159_418_714_200_0.jpg](images/10_159_418_714_200_0.jpg)

Fig. 6. Visualization of soft cluster assignments of 10 users on Amazon dataset. The horizontal axis denotes \( N = {10} \) soft cluster centroids. If two users achieve high scores (i.e., dark color) on the same soft cluster centroids, they tend to have common preference.

From the results shown in Table VI, we find that our NF-NPCDR achieves the highest ratings in all three candidate movies (e.g., 4.57 on Adventure, 4.32 on Romantic, 4.79 on Fantasy). On the contrary, NF-NPCDR (w/o NF) only achieves high score on the Fantasy candidate movie (e.g., 3.11 on Adventure, 3.46 on Romantic, 4.28 on Fantasy). We also note that there are similar observations on CDRNP and PTUPCDR. The above observations indicate that our NF-NPCDR can capture user's personalized multi-interest preference (including Adventure, Romantic and Fantasy), even though the user interacts with fewer Adventure and Romantic books in the source domain. This phenomenon can also be explained by the fact that without the normalizing flow, NF-NPCDR (w/o NF) only captures the user's Fantasy interest that the user interacts with most in the source domain.

## H. Visualization of Preference Pool (RQ3)

In this section, we further visualize the results of soft cluster assignments derived from the interaction between users in \( {\mathcal{T}}_{i} \) and the preference pool \( \mathcal{P} \) . Specifically, we randomly sample ten users from the training tasks \( {\Omega }^{tr} \) , each user would interact with the preference pool to generate the appropriate soft cluster assignments. If two users achieve high scores (i.e., dark color) on the same soft cluster centroids, they tend to have common preference. On the contrary, if the soft cluster assignments of two users are different, they tend to have dissimilar preference.

The visualization results are shown in Fig. 6. we observe that the preference pool proposed in our model can capture the common preference between different users. For example, the second and the seventh clusters of \( {\mathcal{T}}_{2} \) and \( {\mathcal{T}}_{6} \) in Movie \( \rightarrow \) Music are simultaneously assigned the highest probability, which indicates that these two users have similar preference. Meanwhile, the side information also provides proof. We find that these two users both interact with The Lord of the Rings in the source domain. We also have similar findings in other two CDR scenarios. Specifically, we observe that the first and fourth clusters of \( {\mathcal{T}}_{4} \) and \( {\mathcal{T}}_{7} \) in Book \( \rightarrow \) Movie are simultaneously assigned the highest probability, as well as the second and sixth clusters of \( {\mathcal{T}}_{8} \) and \( {\mathcal{T}}_{10} \) in Book \( \rightarrow \) Music, which means that the preference pool can model common preference between different users. Another observation is that users with dissimilar preference are also distinguish well. For example, \( {\mathcal{T}}_{9} \) and \( {\mathcal{T}}_{10} \) in Book \( \rightarrow \) Music have quite different soft cluster assignments, which indicates that these two users have dissimilar preference. From the side information, we find that the interactions of these two users in the source domain do not overlap. Similarly, the soft cluster assignments of \( {\mathcal{T}}_{5} \) and \( {\mathcal{T}}_{10} \) in Book \( \rightarrow \) Movie, as well as \( {\mathcal{T}}_{5} \) and \( {\mathcal{T}}_{7} \) in Book \( \rightarrow \) Music are quite different. This demonstrates that these users have dissimilar preference.

## I. Study of Neural Process (RQ4)

In this section, we further investigate the impact of neural process in NF-NPCDR (w/o NF) and CDRNP [38]. As shown in Fig. 4, we evaluate the performance of NF-NPCDR (w/o NF) and CDRNP with different lengths of support set \( {\mathcal{C}}_{i} \) . We also conduct a case study to investigate the performance of NF-NPCDR (w/o NF) and CDRNP in Fig. VI.

From the results shown in Fig. 4, we observe that NF-NPCDR (w/o NF) consistently outperforms CDRNP with different lengths of support set \( {\mathcal{C}}_{i} \) . This demonstrates that directly transferring the users' personalized preference from source to target domain via neural process can achieve better experimental performance. Another observation is that NF-NPCDR (w/o NF) maintains robust performance with different lengths of support set \( {\mathcal{C}}_{i} \) . In contrast, CDRNP is more sensitive to the quality of \( {\mathcal{C}}_{i} \) . The reasons may be that CDRNP splits the interactions of multiple users into one task. There may exist some users with completely different preference, making the model susceptible to noise information.

From the results shown in Table. VI, we can also see that the ratings on all three candidate movies of NF-NPCDR (w/o NF) are higher than CDRNP. This indicates that directly leverages the neural process to bridge the source and target domains can achieve significant experimental performance.

## J. Study of Normalizing Flow (RQ5)

We conduct a comprehensive comparative studies on the performance and computational costs of the three normalizing flows (i.e., Planar, Radial and RealNVP). Specifically, we explore the NF-NPCDR performance and the training time (seconds) for one epoch with different flow steps \( K \) . From the results shown in Fig. 7, we find that compared with \( w/o\mathrm{{NF}} \) variant, all three normalizing flows could bring performance improvements, which demonstrates the importance of normalizing flow to capture the user's personalized multi-interest preference. Another observation is that NF-NPCDR \( w \) / Planar flow consistently outperforms the other flows in all three CDR scenarios with different flow steps, while their computational costs (i.e., training time) are close. This observation indicates that a brief normalizing flow is sufficient to effectively capture user's multi-interest preference in CDR. Therefore, we choose Planar flow in our experiments, which is effective and efficient.

TABLE VII

Analysis on the GPU usage, parameters and training time for one epoch of PTUPCDR, CDRNP and NF-NPCDR on Amazon dataset.

<table><tr><td>PTUPCDR / CDRNP / NF-NPCDR</td><td>Movie \( \rightarrow \) Music</td><td>Book \( \rightarrow \) Movie</td><td>Book \( \rightarrow \) Music</td></tr><tr><td>#GPU usage</td><td>1.52G / 1.41G / 1.33G</td><td>2.59G / 2.20G / 1.83G</td><td>2.46G / 2.10G / 1.80G</td></tr><tr><td>#Parameters</td><td>25.33M / 18.02M / 11.02M</td><td>84.05M / 66.77M / 38.39M</td><td>80.19M / 65.95M / 37.98M</td></tr><tr><td>#Training time</td><td>307.6s / 56.2s / 177.1s</td><td>583.1s / 98.5s / 364.6s</td><td>288.9s / 43.3s / 168.0s</td></tr></table>

![11_157_390_713_431_0.jpg](images/11_157_390_713_431_0.jpg)

Fig. 7. Performance and training time (seconds) for one epoch of NF-NPCDR under different flow steps \( K \) on Amazon dataset.

We further introspect the results and observe that the performance of NF-NPCDR (all three normalizing flow variants) improves as the number of steps \( K \) increases. This observation indicates that the normalizing flows are able to model a multimodal distribution to capture the user's multi-interest preference with increased steps \( K \) . We also find that while normalizing flows enhance the effectiveness of the neural process, the computational expense also rises with an increase in the number of steps \( K \) . The results shown in Fig. 7 clearly demonstrate that a gradual increase in training time as the steps \( K \) increase. Through the above observations, we set the steps \( K \) of normalizing (Planar) flows to six in Movie \( \rightarrow \) Music but four in Book \( \rightarrow \) Movie and Book \( \rightarrow \) Music. The reason is that, incorporating an excessive number of flows not only slows down computation but also risks overfitting. Thus, selecting a computationally brief flow or using a smaller number of steps \( K \) to further enhance the computational efficiency.

## K. Computational Cost Analysis (RQ6)

We further provide the computational cost analysis of the time and space costs for our NF-NPCDR and the classical method PTUPCDR [14], as well as the SOTA method CDRNP [38], including GPU usage, parameters and training time for one epoch. From the results shown in Table VII, we find that despite the slightly longer training time compared to CDRNP, the efficiency of NF-NPCDR is still in the same order of magnitude as PTUPCDR. Moreover, the GPU usage and the parameters of NF-NPCDR are much lower than those of PTUPCDR and CDRNP, further demonstrating the scalability and feasibility of our personalized multi-interest modeling framework for large-scale deployment. Besides, considering the significant performance improvements of our NF-NPCDR, the computational costs are reasonable and acceptable.

![11_1007_392_551_214_0.jpg](images/11_1007_392_551_214_0.jpg)

Fig. 8. Sensitivity of NF-NPCDR to hyper-parameters \( \lambda \) and \( N \) on Amazon.

## L. Hyper-parameter Analysis

We explore the effect of different hyper-parameter settings to our NF-NPCDR in this section, e.g., \( \lambda \) in Eq. (18) and the number of soft cluster centroids \( N \) in the preference pool \( \mathcal{P} \) . As the results shown in Fig. 8, we observe that in scenario with extensive interactions, such as the Amazon-Book \( \rightarrow \) Amazon-Movie, \( \lambda \) and \( N \) can be assigned a larger value. This may be due to the fact that extensive interactions scenario typically involves more complex common preference between different users, which should be captured by larger \( \lambda \) and \( N \) . In contrast, for the Amazon-Movie \( \rightarrow \) Amazon-Music and Amazon-Book \( \rightarrow \) Amazon-Music, smaller \( \lambda \) and \( N \) can achieve good recommendation performance. We also find that NF-NPCDR maintains robust performance with different values of \( \lambda \) and \( N \) . This can be attributed to our personalized multi-interest modeling framework, which enhances the neural process with the normalizing flow to convert the Gaussian (unimodal) distribution to a multimodal distribution, making our model more robust and stable to the hyper-parameters.

## VI. CONCLUSION

In this paper, we propose a novel personalized multi-interest modeling framework for CDR to cold-start users. We first enhance the neural process with the normalizing flow to convert the Gaussian (unimodal) distribution to a multimodal distribution, which captures the user's personalized multi-interest preference. Then, we propose a common preference encoder with a preference pool to capture the common preference between different users. Furthermore, we introduce a stochastic adaptive decoder to further incorporate both the personalized and common preference for cold-start users, adaptively modulating both preference for better recommendation. Extensive experiments demonstrate that NF-NPCDR consistently outperforms the previous SOTA approaches in five real-world CDR scenarios.

## REFERENCES

[1] S. Rendle, C. Freudenthaler, Z. Gantner, and L. Schmidt-Thieme, "Bpr: Bayesian personalized ranking from implicit feedback," arXiv preprint arXiv:1205.2618, 2012.

[2] A. P. Singh and G. J. Gordon, "Relational learning via collective matrix factorization," in Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining, 2008, pp. 650- 658.

[3] X. He, L. Liao, H. Zhang, L. Nie, X. Hu, and T.-S. Chua, "Neural collaborative filtering," in Proceedings of the 26th international conference on world wide web, 2017, pp. 173-182.

[4] X. Wang, X. He, M. Wang, F. Feng, and T.-S. Chua, "Neural graph collaborative filtering," in Proceedings of the 42nd international ACM SIGIR conference on Research and development in Information Retrieval, 2019, pp. 165-174.

[5] X. Lin, J. Wu, C. Zhou, S. Pan, Y. Cao, and B. Wang, "Task-adaptive neural process for user cold-start recommendation," in Proceedings of the Web Conference 2021, 2021, pp. 1306-1316.

[6] S. Kang, J. Hwang, D. Lee, and H. Yu, "Semi-supervised learning for cross-domain recommendation to cold-start users," in Proceedings of the 28th ACM International Conference on Information and Knowledge Management, 2019, pp. 1563-1572.

[7] T. Man, H. Shen, X. Jin, and X. Cheng, "Cross-domain recommendation: An embedding and mapping approach." in IJCAI, vol. 17, 2017, pp. 2464-2470.

[8] A. Salah, T. B. Tran, and H. Lauw, "Towards source-aligned variational models for cross-domain recommendation," in Proceedings of the 15th ACM Conference on Recommender Systems, 2021, pp. 176-186.

[9] J. Shi and Q. Wang, "Cross-domain variational autoencoder for recommender systems," in 2019 IEEE 11th International Conference on Advanced Infocomm Technology (ICAIT). IEEE, 2019, pp. 67-72.

[10] Y. Zhu, K. Ge, F. Zhuang, R. Xie, D. Xi, X. Zhang, L. Lin, and Q. He, "Transfer-meta framework for cross-domain recommendation to cold-start users," in Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2021, pp. 1813-1817.

[11] F. Zhu, Y. Wang, C. Chen, G. Liu, M. Orgun, and J. Wu, "A deep framework for cross-domain and cross-system recommendations," in Proceedings of the 27th International Joint Conference on Artificial Intelligence, 2018, pp. 3711-3717.

[12] Y. Bi, L. Song, M. Yao, Z. Wu, J. Wang, and J. Xiao, "Dcdir: A deep cross-domain recommendation system for cold start users in insurance domain," in Proceedings of the 43rd international ACM SIGIR conference on research and development in information retrieval, 2020, pp. 1661-1664.

[13] \\", "A heterogeneous information network based cross domain insurance recommendation system for cold start users," in Proceedings of the 43rd international ACM SIGIR conference on research and development in information retrieval, 2020, pp. 2211-2220.

[14] Y. Zhu, Z. Tang, Y. Liu, F. Zhuang, R. Xie, X. Zhang, L. Lin, and Q. He, "Personalized transfer of user preferences for cross-domain recommendation," in Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining, 2022, pp. 1507-1515.

[15] R. Guan, H. Pang, F. Giunchiglia, Y. Liang, and X. Feng, "Cross-domain meta-learner for cold-start recommendation," IEEE Transactions on Knowledge and Data Engineering, 2022.

[16] Y. Cen, J. Zhang, X. Zou, C. Zhou, H. Yang, and J. Tang, "Controllable multi-interest framework for recommendation," in Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2020, pp. 2942-2951.

[17] G. Chen, X. Zhang, Y. Zhao, C. Xue, and J. Xiang, "Exploring periodicity and interactivity in multi-interest framework for sequential recommendation," Proceedings of the 30th International Joint Conference on Artificial Intelligence, 2021.

[18] M. Garnelo, J. Schwarz, D. Rosenbaum, F. Viola, D. J. Rezende, S. Eslami, and Y. W. Teh, "Neural processes," arXiv preprint arXiv:1807.01622, 2018.

[19] G. Papamakarios, E. Nalisnick, D. J. Rezende, S. Mohamed, and B. Lakshminarayanan, "Normalizing flows for probabilistic modeling and inference," The Journal of Machine Learning Research, vol. 22, no. 1, pp. 2617-2680, 2021.

[20] E. G. Tabak and C. V. Turner, "A family of nonparametric density estimation algorithms," Communications on Pure and Applied Mathematics, vol. 66, no. 2, pp. 145-164, 2013.

[21] S. Li, L. Yao, S. Mu, W. X. Zhao, Y. Li, T. Guo, B. Ding, and J.-R. Wen, "Debiasing learning based cross-domain recommendation," in Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, 2021, pp. 3190-3199.

[22] W. Fan, T. Derr, X. Zhao, Y. Ma, H. Liu, J. Wang, J. Tang, and Q. Li, "Attacking black-box recommendations via copying cross-domain user profiles," in 2021 IEEE 37th International Conference on Data Engineering (ICDE). IEEE, 2021, pp. 1583-1594.

[23] P. Li and A. Tuzhilin, "Dual metric learning for effective and efficient cross-domain recommendations," IEEE Transactions on Knowledge and Data Engineering, vol. 35, no. 1, pp. 321-334, 2021.

[24] C. Zhao, H. Zhao, X. Li, M. He, J. Wang, and J. Fan, "Cross-domain recommendation via progressive structural alignment," IEEE Transactions on Knowledge and Data Engineering, 2023.

[25] G. Hu, Y. Zhang, and Q. Yang, "Conet: Collaborative cross networks for cross-domain recommendation," in Proceedings of the 27th ACM international conference on information and knowledge management, 2018, pp. 667-676.

[26] B. Li, Q. Yang, and X. Xue, "Can movies and books collaborate? cross-domain collaborative filtering for sparsity reduction," in Twenty-First international joint conference on artificial intelligence, 2009.

[27] J. Cao, X. Lin, X. Cong, J. Ya, T. Liu, and B. Wang, "Disencdr: Learning disentangled representations for cross-domain recommendation," in Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2022, pp. 267-277.

[28] W. Liu, X. Zheng, J. Su, L. Zheng, C. Chen, and M. Hu, "Contrastive proxy kernel stein path alignment for cross-domain cold-start recommendation," IEEE Transactions on Knowledge and Data Engineering, 2023.

[29] J. Cao, J. Sheng, X. Cong, T. Liu, and B. Wang, "Cross-domain recommendation to cold-start users via variational information bottleneck," in 2022 IEEE 38th International Conference on Data Engineering (ICDE). IEEE, 2022, pp. 2209-2223.

[30] C. Sun, J. Gu, B. Hu, X. Dong, H. Li, L. Cheng, and L. Mo, "Remit: reinforced multi-interest transfer for cross-domain recommendation," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 8, 2023, pp. 9900-9908.

[31] M. Garnelo, D. Rosenbaum, C. Maddison, T. Ramalho, D. Saxton, M. Shanahan, Y. W. Teh, D. Rezende, and S. A. Eslami, "Conditional neural processes," in International conference on machine learning. PMLR, 2018, pp. 1704-1713.

[32] H. Kim, A. Mnih, J. Schwarz, M. Garnelo, A. Eslami, D. Rosenbaum, O. Vinyals, and Y. W. Teh, "Attentive neural processes," arXiv preprint arXiv:1901.05761, 2019.

[33] J. Shen, X. Zhen, Q. Wang, and M. Worring, "Episodic multi-task learning with heterogeneous neural processes," Advances in Neural Information Processing Systems, vol. 36, pp. 75214-75228, 2023.

[34] J. Wang, D. Massiceti, X. Hu, V. Pavlovic, and T. Lukasiewicz, "Np-semiseg: when neural processes meet semi-supervised semantic segmentation," in International Conference on Machine Learning. PMLR, 2023, pp. 36138-36156.

[35] J. Du, Z. Ye, B. Guo, Z. Yu, and L. Yao, "Idnp: Interest dynamics modeling using generative neural processes for sequential recommendation," in Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining, 2023, pp. 481-489.

[36] X. Lin, C. Zhou, J. Wu, L. Zou, S. Pan, Y. Cao, B. Wang, S. Wang, and D. Yin, "Towards flexible and adaptive neural process for cold-start recommendation," IEEE Transactions on Knowledge and Data Engineering, 2023.

[37] H. Liu, L. Jing, D. Yu, M. Zhou, and M. Ng, "Learning intrinsic and extrinsic intentions for cold-start recommendation with neural stochastic processes," in Proceedings of the 30th ACM International Conference on Multimedia, 2022, pp. 491-500.

[38] X. Li, J. Sheng, J. Cao, W. Zhang, Q. Li, and T. Liu, "Cdrnp: Cross-domain recommendation to cold-start users via neural process," in Proceedings of the 17th ACM International Conference on Web Search and Data Mining, 2024, pp. 378-386.

[39] I. Kobyzev, S. J. Prince, and M. A. Brubaker, "Normalizing flows: An introduction and review of current methods," IEEE transactions on pattern analysis and machine intelligence, vol. 43, no. 11, pp. 3964- 3979, 2020.

[40] D. Rezende and S. Mohamed, "Variational inference with normalizing flows," in International conference on machine learning. PMLR, 2015, pp. 1530-1538.

[41] L. Dinh, J. Sohl-Dickstein, and S. Bengio, "Density estimation using real nvp," arXiv preprint arXiv:1605.08803, 2016.

[42] G. Papamakarios, T. Pavlakou, and I. Murray, "Masked autoregressive flow for density estimation," Advances in neural information processing systems, vol. 30, 2017.

[43] B. Mazoure, T. Doan, A. Durand, J. Pineau, and R. D. Hjelm, "Leveraging exploration in off-policy algorithms via normalizing flows," in Conference on Robot Learning. PMLR, 2020, pp. 430-444.

[44] A. Touati, H. Satija, J. Romoff, J. Pineau, and P. Vincent, "Randomized value functions via multiplicative normalizing flows," in Uncertainty in Artificial Intelligence. PMLR, 2020, pp. 422-432.

[45] K. Madhawa, K. Ishiguro, K. Nakago, and M. Abe, "Graphnvp: An invertible flow model for generating molecular graphs," arXiv preprint arXiv:1905.11600, 2019.

[46] L. Luo, Y.-F. Li, G. Haffari, and S. Pan, "Normalizing flow-based neural process for few-shot knowledge graph completion," arXiv preprint arXiv:2304.08183, 2023.

[47] D. P. Kingma and M. Welling, "Auto-encoding variational bayes," arXiv preprint arXiv:1312.6114, 2013.

[48] A. Mnih and K. Gregor, "Neural variational inference and learning in belief networks," in International Conference on Machine Learning. PMLR, 2014, pp. 1791-1799.

[49] D. Rezende and S. Mohamed, "Variational inference with normalizing flows," in International conference on machine learning. PMLR, 2015, pp. 1530-1538.

[50] B. Øksendal and B. Øksendal, Stochastic differential equations. Springer, 2003.

[51] J. Xie, R. Girshick, and A. Farhadi, "Unsupervised deep embedding for clustering analysis," in International conference on machine learning. PMLR, 2016, pp. 478-487.

[52] L. Van der Maaten and G. Hinton, "Visualizing data using t-sne." Journal of machine learning research, vol. 9, no. 11, 2008.

[53] E. Perez, F. Strub, H. De Vries, V. Dumoulin, and A. Courville, "Film: Visual reasoning with a general conditioning layer," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32, no. 1, 2018.

[54] X. Fan, J. Lian, W. X. Zhao, Z. Liu, C. Li, and X. Xie, "Ada-ranker: A data distribution adaptive ranking paradigm for sequential recommendation," in Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2022, pp. 1599-1610.

[55] Y. Cen, J. Zhang, X. Zou, C. Zhou, H. Yang, and J. Tang, "Controllable multi-interest framework for recommendation," in Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2020, pp. 2942-2951.

[56] W. Guo, C. Zhang, Z. He, J. Qin, H. Guo, B. Chen, R. Tang, X. He, and R. Zhang, "Miss: Multi-interest self-supervised learning framework for click-through rate prediction," in 2022 IEEE 38th international conference on data engineering (ICDE). IEEE, 2022, pp. 727-740.

[57] Y. Tian, J. Chang, Y. Niu, Y. Song, and C. Li, "When multi-level meets multi-interest: A multi-grained neural model for sequential recommendation," in Proceedings of the 45th international ACM SIGIR conference on research and development in information retrieval, 2022, pp. 1632- 1641.

[58] Z. Wang and Y. Shen, "Incremental learning for multi-interest sequential recommendation," in 2023 IEEE 39th International Conference on Data Engineering (ICDE). IEEE, 2023, pp. 1071-1083.

[59] S. Zhang, L. Yang, D. Yao, Y. Lu, F. Feng, Z. Zhao, T.-S. Chua, and F. Wu, "Re4: Learning to re-contrast, re-attend, re-construct for multi-interest recommendation," in Proceedings of the ACM Web Conference 2022, 2022, pp. 2216-2226.

[60] Y. Zheng, G. Wang, Y. Liu, and L. Lin, "Diversity matters: User-centric multi-interest learning for conversational movie recommendation," in Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 9515-9524.

[61] C. Zhao, C. Li, R. Xiao, H. Deng, and A. Sun, "Catn: Cross-domain recommendation for cold-start users via aspect transfer network," in Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, 2020, pp. 229- 238.

[62] W. Fu, Z. Peng, S. Wang, Y. Xu, and J. Li, "Deeply fusing reviews and contents for cold start users in cross-domain recommendation systems," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 33, no. 01, 2019, pp. 94-101.

[63] C.-K. Hsieh, L. Yang, Y. Cui, T.-Y. Lin, S. Belongie, and D. Estrin, "Collaborative metric learning," in Proceedings of the 26th international conference on world wide web, 2017, pp. 193-201.

[64] L. Hu, J. Cao, G. Xu, L. Cao, Z. Gu, and C. Zhu, "Personalized recommendation via cross-domain triadic factorization," in Proceedings of the 22nd international conference on World Wide Web, 2013, pp. 595-606.

[65] A. Mnih and R. R. Salakhutdinov, "Probabilistic matrix factorization," Advances in neural information processing systems, vol. 20, 2007.

[66] T. Wang, F. Zhuang, Z. Zhang, D. Wang, J. Zhou, and Q. He, "Low-dimensional alignment for cross-domain recommendation," in Proceedings of the 30th ACM international conference on information & knowledge management, 2021, pp. 3508-3512.

[67] C. Li, M. Zhao, H. Zhang, C. Yu, L. Cheng, G. Shu, B. Kong, and D. Niu, "Recguru: Adversarial learning of generalized user representations for cross-domain recommendation," in Proceedings of the fifteenth ACM international conference on web search and data mining, 2022, pp. 571-581.

[68] H. Bharadhwaj, "Meta-learning for user cold-start recommendation," in 2019 International Joint Conference on Neural Networks (IJCNN). IEEE, 2019, pp. 1-8.

[69] M. Dong, F. Yuan, L. Yao, X. Xu, and L. Zhu, "Mamo: Memory-augmented meta-optimization for cold-start recommendation," in Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, 2020, pp. 688-697.

[70] H. Lee, J. Im, S. Jang, H. Cho, and S. Chung, "Melu: Meta-learned user preference estimator for cold-start recommendation," in Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2019, pp. 1073-1082.