<h1 align="center">✨✨✨ Empowering MLLMs with External Tools: A Survey ✨✨✨</h1>

<p align="center"><em>Curated list of empowering Multimodal Large Language Models (MLLMs) with external tools, from the perspectives of </em><br><strong>Data, Tasks, and Evaluation. </strong> (🚧 Under Construction)</p>

<p align="center">
    <a href="https://arxiv.org/abs/2502.14881"><img src="https://img.shields.io/badge/arXiv-2502.14881-b31b1b.svg" alt="arXiv Badge"></a>
    <a href="https://awesome.re"><img src="https://awesome.re/badge.svg" alt="Awesome Badge"></a>
    <a href="https://creativecommons.org/licenses/by-nc/4.0/"><img src="https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg" alt="License Badge"></a>
    <a href="[https://github.com/Lackel/Awesome-Tools-for-MLLMs](https://github.com/Lackel/Awesome-Tools-for-MLLMs)"><img src="https://img.shields.io/github/stars/Lackel/Awesome-Tools-for-MLLMs?style=social" alt="GitHub stars"></a>
</p>

<h2 align="center">🎉 Overview 🎉</h2>

![](/assets/overview.png)

<h2 align="center">🚀 Abstract 🚀</h2>

<p align="justify">By integrating the perception capabilities of multimodal encoders with the generative power of Large Language Models (LLMs), Multimodal Large Language Models (MLLMs), exemplified by GPT-4V, have achieved great success in various multimodal tasks, pointing toward a promising pathway to artificial general intelligence.</p>

<p align="justify">
Despite this progress, the limited quality of multimodal data, poor performance on many complex downstream tasks, and inadequate evaluation protocols continue to hinder the reliability and broader applicability of MLLMs across diverse domains.
Inspired by the human ability to leverage external tools for enhanced reasoning and problem-solving, augmenting MLLMs with external tools (e.g., APIs, expert models, and knowledge bases) offers a promising strategy to overcome these challenges.
</p>

<p align="justify">
In this paper, we present a comprehensive survey on leveraging external tools to enhance MLLM performance. Our discussion is structured along four key dimensions about external tools: (1) how they can facilitate the acquisition and annotation of high-quality multimodal data; (2) how they can assist in improving MLLM performance on challenging downstream tasks; (3) how they enable comprehensive and accurate evaluation of MLLMs; and (4) the current limitations and future directions of tool-augmented MLLMs.
</p>

<p align="justify">
Through this survey, we aim to underscore the transformative potential of external tools in advancing MLLM capabilities, offering a forward-looking perspective on their development and applications.
</p>

<h2 align="center"> 📜 Table of Contents 📜</h2>

- [Awesome Papers](#awesome-papers)
  - [Data](#data)
      - [Data Collection](#collection)
      - [Data Synthesis](#synthesis)
      - [Data Annotation](#annotation)
      - [Data Cleaning](#cleaning)
  - [Tasks](#tasks)
      - [Multimodal Retrieval Augmented Generation](#mrag)
      - [Multimodal Reasoning](#mr)
      - [Multimodal Hallucination](#mh)
      - [Multimodal Safety](#ms)
      - [Multimodal Agents](#ma)
      - [Video Perception](#vp)
  - [Evaluation](#evaluation)
      - [Keyword Extraction](#ke)
      - [Embedding-based Evaluation](#ee)
      - [MLLM-based Evaluation](#me)
      - [Evaluation Platform](#ep)
  - [Related Surveys](#rs)

<h2 align="center" id="awesome-papers"> 👑 Awesome Papers 👑</h2>

<h3 align="center" id="data"> 🥑Data </h3>

<h4 id="collection"> 1. Data Collection </h4>
<ul>
<li><a href="https://arxiv.org/abs/2402.14683"><b>Visual hallucinations of multi-modal large language models</b></a></li>
<li><a href="https://aclanthology.org/2024.emnlp-main.904/"><b>The Instinctive Bias: Spurious Images lead to Illusion in MLLMs</b></a></li>
<li><a href="https://arxiv.org/abs/2401.06209"><b>Eyes wide shut? exploring the visual shortcomings of multimodal llms</b></a></li>
<li><a href="https://aclanthology.org/P18-1238"><b>Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning</b></a></li>
<li><a href="https://dl.acm.org/doi/10.1145/1809028.1806638"><b>FlumeJava: easy, efficient data-parallel pipelines</b></a></li>
<li><a href="https://arxiv.org/abs/2209.06794"><b>Pali: A jointly-scaled multilingual language-image model</b></a></li>
<li><a href="https://arxiv.org/abs/2210.08402"><b>Laion-5b: An open large-scale dataset for training next generation image-text models</b></a></li>
<li><a href="https://arxiv.org/abs/2304.14108"><b>Datacomp: In search of the next generation of multimodal datasets</b></a></li>
<li><a href="https://arxiv.org/abs/2102.05918"><b>Scaling up visual and vision-language representation learning with noisy text supervision</b></a></li>
<li><a href="https://arxiv.org/abs/2304.06939"><b>Multimodal c4: An open, billion-scale corpus of images interleaved with text</b></a></li>
<li><a href="https://arxiv.org/abs/2202.06767"><b>Wukong: A 100 million large-scale chinese cross-modal pre-training benchmark</b></a></li>
<li><a href="https://arxiv.org/abs/2103.01913"><b>Wit: Wikipedia-based image text dataset for multimodal multilingual machine learning</b></a></li>
<li><a href="https://arxiv.org/abs/2102.08981"><b>Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts</b></a></li>
<li><a href="https://arxiv.org/abs/2111.11431"><b>Redcaps: Web-curated image-text data created by the people, for the people</b></a></li>
<li><a href="https://aclanthology.org/2024.acl-long.775"><b>Multimodal arxiv: A dataset for improving scientific comprehension of large vision-language models</b></a></li>
<li><a href="https://arxiv.org/abs/1612.00837"><b>Making the v in vqa matter: Elevating the role of image understanding in visual question answering</b></a></li>
<li><a href="https://arxiv.org/abs/1711.06475"><b>Ai challenger: A large-scale dataset for going deeper in image understanding</b></a></li>
<li><a href="https://dl.acm.org/doi/10.5555/2986459.2986587"><b>Im2text: Describing images using 1 million captioned photographs</b></a></li>
</ul>

<h4 id="synthesis"> 2. Data Synthesis </h4>
<ul>
<li><a href="https://arxiv.org/abs/2410.09962"><b>Longhalqa: Long-context hallucination evaluation for multimodal large language models</b></a></li>
<li><a href="https://arxiv.org/abs/2406.10900"><b>Autohallusion: Automatic generation of hallucination benchmarks for vision-language models</b></a></li>
<li><a href="https://arxiv.org/abs/2405.05256"><b>Throne: An object-based hallucination benchmark for the free-form generations of large vision-language models</b></a></li>
<li><a href="https://arxiv.org/abs/2403.11116"><b>Phd: A chatgpt-prompted visual hallucination evaluation dataset</b></a></li>
<li><a href="https://arxiv.org/abs/2403.08542"><b>AIGCs confuse AI too: Investigating and explaining synthetic image-induced hallucinations in large vision-language models</b></a></li>
<li><a href="https://arxiv.org/abs/2402.14683"><b>Visual hallucinations of multi-modal large language models</b></a></li>
<li><a href="https://arxiv.org/abs/2402.13220"><b>How easy is it to fool your multimodal llms? an empirical analysis on deceptive prompts</b></a></li>
<li><a href="https://arxiv.org/abs/2312.03631"><b>Mitigating open-vocabulary caption hallucinations</b></a></li>
<li><a href="https://aclanthology.org/2024.emnlp-main.904/"><b>The Instinctive Bias: Spurious Images lead to Illusion in MLLMs</b></a></li>
<li><a href="https://arxiv.org/abs/2312.01701"><b>Mitigating fine-grained hallucination by fine-tuning large vision-language models with caption rewrites</b></a></li>
<li><a href="https://arxiv.org/abs/2310.05338"><b>Negative object presence evaluation (nope) to measure object hallucination in vision-language models</b></a></li>
<li><a href="https://arxiv.org/abs/2306.14565"><b>Mitigating hallucination in large multi-modal models via robust instruction tuning</b></a></li>
<li><a href="https://arxiv.org/abs/2309.02301"><b>Ciem: Contrastive instruction evaluation method for better instruction tuning</b></a></li>
<li><a href="https://arxiv.org/abs/2407.00569"><b>Investigating and mitigating the multimodal hallucination snowballing in large vision-language models</b></a></li>
<li><a href="https://arxiv.org/abs/2312.10665"><b>Silkie: Preference distillation for large visual language models</b></a></li>
<li><a href="https://aclanthology.org/2024.acl-long.775"><b>Multimodal arxiv: A dataset for improving scientific comprehension of large vision-language models</b></a></li>
<li><a href="https://arxiv.org/abs/1801.08163"><b>Dvqa: Understanding data visualizations via question answering</b></a></li>
<li><a href="https://arxiv.org/abs/2406.09121"><b>Mmrel: A relation understanding dataset and benchmark in the mllm era</b></a></li>
<li><a href="https://arxiv.org/abs/2304.08485"><b>Visual Instruction Tuning</b></a></li>
<li><a href="https://aclanthology.org/2024.emnlp-main.1016/"><b>Investigating and mitigating object hallucinations in pretrained vision-language (clip) models</b></a></li>
</ul>    

<h4 id="annotation"> 3. Data Annotation </h4>
<ul>
<li><a href="https://arxiv.org/abs/2406.14056"><b>Vga: Vision gui assistant-minimizing hallucinations through image-centric fine-tuning</b></a></li>
<li><a href="https://arxiv.org/abs/2404.02904"><b>ALOHa: A new measure for hallucination in captioning models</b></a></li>
<li><a href="https://arxiv.org/abs/2403.11116"><b>Phd: A chatgpt-prompted visual hallucination evaluation dataset</b></a></li>
<li><a href="https://arxiv.org/abs/2402.15721"><b>Hal-eval: A universal and fine-grained hallucination evaluation framework for large vision language models</b></a></li>
<li><a href="https://arxiv.org/abs/2402.03190"><b>Unified hallucination detection for multimodal large language models</b></a></li>
<li><a href="https://arxiv.org/abs/2312.03631"><b>Mitigating open-vocabulary caption hallucinations</b></a></li>
<li><a href="https://arxiv.org/abs/2312.02219"><b>Behind the magic, merlim: Multi-modal evaluation benchmark for large image-language models</b></a></li>
<li><a href="https://arxiv.org/abs/2311.16479"><b>Mitigating hallucination in visual language models with visual supervision</b></a></li>
<li><a href="https://arxiv.org/abs/2406.09121"><b>Mmrel: A relation understanding dataset and benchmark in the mllm era</b></a></li>
<li><a href="https://arxiv.org/abs/2310.01779"><b>HallE-Control: controlling object hallucination in large multimodal models</b></a></li>
<li><a href="https://arxiv.org/abs/2308.15126"><b>Evaluation and analysis of hallucination in large vision-language models</b></a></li>
<li><a href="https://arxiv.org/abs/2305.10355"><b>Evaluating object hallucination in large vision-language models</b></a></li>
<li><a href="https://arxiv.org/abs/2404.16375"><b>List items one by one: A new data source and learning paradigm for multimodal llms</b></a></li>
<li><a href="https://arxiv.org/abs/2308.06394"><b>Detecting and preventing hallucinations in large vision language models</b></a></li>
<li><a href="https://arxiv.org/abs/2412.20622"><b>Towards a Systematic Evaluation of Hallucinations in Large-Vision Language Models</b></a></li>
<li><a href="https://papers.nips.cc/paper/4470-im2text-describing-images-using-1-million-captioned-photographs"><b>Im2text: Describing images using 1 million captioned photographs</b></a></li>
<li><a href="https://arxiv.org/abs/2209.06794"><b>Pali: A jointly-scaled multilingual language-image model</b></a></li>
</ul>

<h4 id="cleaning"> 4. Data Cleaning </h4>
<ul>
<li><a href="https://arxiv.org/abs/2406.16338"><b>Videohallucer: Evaluating intrinsic and extrinsic hallucinations in large video-language models</b></a></li>
<li><a href="https://arxiv.org/abs/2403.15952"><b>Illusionvqa: A challenging optical illusion dataset for vision language models</b></a></li>
<li><a href="https://arxiv.org/abs/2403.08542"><b>AIGCs confuse AI too: Investigating and explaining synthetic image-induced hallucinations in large vision-language models</b></a></li>
<li><a href="https://arxiv.org/abs/2306.14565"><b>Mitigating hallucination in large multi-modal models via robust instruction tuning</b></a></li>
<li><a href="https://arxiv.org/abs/2405.15683"><b>Visual description grounding reduces hallucinations and boosts reasoning in lvlms</b></a></li>
<li><a href="https://arxiv.org/abs/2412.03735"><b>Vidhalluc: Evaluating temporal hallucinations in multimodal large language models for video understanding</b></a></li>
<li><a href="https://aclanthology.org/P18-1238/"><b>Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning</b></a></li>
<li><a href="https://arxiv.org/abs/2210.08402"><b>Laion-5b: An open large-scale dataset for training next generation image-text models</b></a></li>
<li><a href="https://arxiv.org/abs/2304.14108"><b>Datacomp: In search of the next generation of multimodal datasets</b></a></li>
<li><a href="https://github.com/kakaobrain/coyo-dataset"><b>COYO-700M: Image-Text Pair Dataset</b></a></li>
<li><a href="https://arxiv.org/abs/2306.16527"><b>Obelics: An open web-scale filtered dataset of interleaved image-text documents</b></a></li>
<li><a href="https://arxiv.org/abs/2304.06939"><b>Multimodal c4: An open, billion-scale corpus of images interleaved with text</b></a></li>
<li><a href="https://arxiv.org/abs/2202.06767"><b>Wukong: A 100 million large-scale chinese cross-modal pre-training benchmark</b></a></li>
<li><a href="https://arxiv.org/abs/2103.01913"><b>Wit: Wikipedia-based image text dataset for multimodal multilingual machine learning</b></a></li>
<li><a href="https://arxiv.org/abs/2102.08981"><b>Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts</b></a></li>
<li><a href="https://arxiv.org/abs/1801.08163"><b>Dvqa: Understanding data visualizations via question answering</b></a></li>
<li><a href="https://arxiv.org/abs/2111.11431"><b>Redcaps: Web-curated image-text data created by the people, for the people</b></a></li>
<li><a href="https://aclanthology.org/2024.acl-long.775"><b>Multimodal arxiv: A dataset for improving scientific comprehension of large vision-language models</b></a></li>
</ul>




<h3 id="tasks" align="center"> 🫐Tasks </h3>

<h4 id="mrag"> 1. Multimodal Retrieval Augmented Generation </h4>
<h5> Knowledge Retrieval </h5>
<ul>
<li><a href="https://arxiv.org/abs/2102.05918"><b>Scaling up visual and vision-language representation learning with noisy text supervision</b></a></li>
<li><a href="https://arxiv.org/abs/2103.00020"><b>Learning transferable visual models from natural language supervision</b></a></li>
<li><a href="https://arxiv.org/abs/2201.12086"><b>Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation</b></a></li>
<li><a href="https://arxiv.org/abs/2412.16855"><b>GME: Improving Universal Multimodal Retrieval by Multimodal LLMs</b></a></li>
<li><a href="https://arxiv.org/abs/2112.09118"><b>Unsupervised dense information retrieval with contrastive learning</b></a></li>
<li><a href="https://arxiv.org/abs/2302.03084"><b>Pic2word: Mapping pictures to words for zero-shot composed image retrieval</b></a></li>
<li><a href="https://aclanthology.org/2024.findings-acl.771/"><b>Xl-headtags: Leveraging multimodal retrieval augmentation for the multilingual generation of news headlines and tags</b></a></li>
<li><a href="https://arxiv.org/abs/2407.12735"><b>Echosight: Advancing visual-language models with wiki knowledge</b></a></li>
<li><a href="https://arxiv.org/abs/2403.10153"><b>Improving medical multi-modal contrastive learning with expert annotations</b></a></li>
<li><a href="https://arxiv.org/abs/2502.01549"><b>VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context Videos</b></a></li>
<li><a href="https://openaccess.thecvf.com/content/CVPR2025/papers/Ma_DrVideo_Document_Retrieval_Based_Long_Video_Understanding_CVPR_2025_paper.pdf"><b>DrVideo: Document Retrieval Based Long Video Understanding</b></a></li>
<li><a href="https://www.ijcai.org/proceedings/2024/136"><b>Contrastive Transformer Cross-Modal Hashing for Video-Text Retrieval</b></a></li>
<li><a href="https://aclanthology.org/2024.emnlp-main.559/"><b>OmAgent: A Multi-modal Agent Framework for Complex Video Understanding with Task Divide-and-Conquer</b></a></li>
<li><a href="https://arxiv.org/abs/2412.16500"><b>Speech Retrieval-Augmented Generation without Automatic Speech Recognition</b></a></li>
<li><a href="https://arxiv.org/abs/2412.14457"><b>VISA: Retrieval Augmented Generation with Visual Source Attribution</b></a></li>
<li><a href="https://arxiv.org/abs/2309.09836"><b>Recap: Retrieval-Augmented Audio Captioning</b></a></li>
<li><a href="https://arxiv.org/abs/2502.14727"><b>WavRAG: Audio-Integrated Retrieval Augmented Generation for Spoken Dialogue Models</b></a></li>
<li><a href="https://arxiv.org/abs/2411.05141"><b>Audiobox TTA-RAG: Improving Zero-Shot and Few-Shot Text-To-Audio with Retrieval-Augmented Generation</b></a></li>
</ul>

<h5> Knowledge Reranking </h5>
<ul>
<li><a href="https://arxiv.org/abs/2407.00978"><b>Hybrid RAG-Empowered Multi-Modal LLM for Secure Data Management in Internet of Medical Things: A Diffusion-Based Contract Approach</b></a></li>
<li><a href="https://arxiv.org/abs/2404.12866"><b>How does the textual information affect the retrieval of multimodal in-context learning?</b></a></li>
<li><a href="https://aclanthology.org/2024.emnlp-main.62/"><b>Rule: Reliable multimodal rag for factuality in medical vision language models</b></a></li>
<li><a href="https://arxiv.org/abs/2303.00534"><b>Ramm: Retrieval-augmented biomedical visual question answering with multi-modal pre-training</b></a></li>
<li><a href="https://arxiv.org/abs/2501.04695"><b>Re-ranking the Context for Multimodal Retrieval Augmented Generation</b></a></li>
<li><a href="https://arxiv.org/abs/2501.03995"><b>Rag-check: Evaluating multimodal retrieval augmented generation performance</b></a></li>
<li><a href="https://arxiv.org/abs/2411.15041"><b>mR2AG: Multimodal Retrieval-Reflection-Augmented Generation for Knowledge-Based VQA</b></a></li>
<li><a href="https://aclanthology.org/2024.emnlp-industry.75/"><b>OMG-QA: Building Open-Domain Multi-Modal Generative Question Answering Systems</b></a></li>
<li><a href="https://arxiv.org/abs/2401.00789"><b>Retrieval-augmented egocentric video captioning</b></a></li>
<li><a href="https://dl.acm.org/doi/10.1145/3626772.3657740"><b>LDRE: LLM-based Divergent Reasoning and Ensemble for Zero-Shot Composed Image Retrieval</b></a></li>
<li><a href="https://arxiv.org/abs/2412.16855"><b>GME: Improving Universal Multimodal Retrieval by Multimodal LLMs</b></a></li>
<li><a href="https://ieeexplore.ieee.org/document/10535103/"><b>UniRaG: Unification, Retrieval, and Generation for Multimodal Question Answering With Pre-Trained Language Models</b></a></li>
<li><a href="https://arxiv.org/abs/2411.02571"><b>Mm-embed: Universal multimodal retrieval with multimodal llms</b></a></li>
<li><a href="https://arxiv.org/abs/2501.00332"><b>MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation</b></a></li>
<li><a href="https://arxiv.org/abs/2407.12735"><b>EchoSight: Advancing Visual-Language Models with Wiki Knowledge</b></a></li>
</ul>

<h5> Knowledge Utilization </h5>
<ul>
<li><a href="https://arxiv.org/abs/2211.12561"><b>Retrieval-augmented multimodal language modeling</b></a></li>
<li><a href="https://arxiv.org/abs/2407.15346"><b>Knowledge Acquisition Disentanglement for Knowledge-based Visual Question Answering with Large Language Models</b></a></li>
<li><a href="https://ieeexplore.ieee.org/document/10535103/"><b>UniRaG: Unification, Retrieval, and Generation for Multimodal Question Answering With Pre-Trained Language Models</b></a></li>
<li><a href="https://arxiv.org/abs/2405.20834"><b>Retrieval meets reasoning: Even high-school textbook knowledge benefits multimodal reasoning</b></a></li>
<li><a href="https://arxiv.org/abs/2412.10704"><b>VisDoM: Multi-Document QA with Visually Rich Elements Using Multimodal Retrieval-Augmented Generation</b></a></li>
<li><a href="https://arxiv.org/abs/2410.11321"><b>Self-adaptive Multimodal Retrieval-Augmented Generation</b></a></li>
<li><a href="https://arxiv.org/abs/2404.12065"><b>Ragar, your falsehood radar: Rag-augmented reasoning for political fact-checking using multimodal large language models</b></a></li>
<li><a href="https://www.techrxiv.org/users/833309/articles/1229891-iterative-retrieval-augmentation-for-multi-modal-knowledge-integration-and-generation"><b>Iterative Retrieval Augmentation for Multi-Modal Knowledge Integration and Generation</b></a></li>
<li><a href="https://arxiv.org/abs/2502.01549"><b>VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context Videos</b></a></li>
<li><a href="https://arxiv.org/abs/2404.15406"><b>Wiki-LLaVA: Hierarchical Retrieval-Augmented Generation for Multimodal LLMs</b></a></li>
<li><a href="https://arxiv.org/abs/2411.16863"><b>Augmenting multimodal llms with self-reflective tokens for knowledge-based visual question answering</b></a></li>
<li><a href="https://arxiv.org/abs/2410.13085"><b>Mmed-rag: Versatile multimodal rag system for medical vision language models</b></a></li>
<li><a href="https://arxiv.org/abs/2501.01120"><b>Retrieval-Augmented Dynamic Prompt Tuning for Incomplete Multimodal Learning</b></a></li>
<li><a href="https://arxiv.org/abs/2411.15041"><b>mR2AG: Multimodal Retrieval-Reflection-Augmented Generation for Knowledge-Based VQA</b></a></li>
<li><a href="https://aclanthology.org/2024.emnlp-main.62/"><b>Rule: Reliable multimodal rag for factuality in medical vision language models</b></a></li>
</ul>

<h4 id="mr"> 2. Multimodal Reasoning </h4>
<h5> Image Reasoning </h5>
<ul>
<li><a href="https://arxiv.org/abs/2506.09736"><b>Vision Matters: Simple Visual Perturbations Can Boost Multimodal Math Reasoning</b></a></li>
<li><a href="https://arxiv.org/abs/2501.05452"><b>ReFocus: Visual Editing as a Chain of Thought for Structured Image Understanding</b></a></li>
<li><a href="https://arxiv.org/abs/2504.05599"><b>Skywork r1v: Pioneering multimodal reasoning with chain-of-thought</b></a></li>
<li><a href="https://arxiv.org/abs/2503.05255"><b>Cmmcot: Enhancing complex multi-image comprehension via multi-modal chain-of-thought and memory augmentation</b></a></li>
<li><a href="https://arxiv.org/abs/2503.05236"><b>Unified reward model for multimodal understanding and generation</b></a></li>
<li><a href="https://arxiv.org/abs/2502.10391"><b>Mm-rlhf: The next step forward in multimodal llm alignment</b></a></li>
<li><a href="https://arxiv.org/abs/2503.10615"><b>R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization</b></a></li>
<li><a href="https://arxiv.org/html/2411.10440v1"><b>Llava-o1: Let vision language models reason step-by-step</b></a></li>
<li><a href="https://arxiv.org/abs/2403.16999"><b>Visual cot: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning</b></a></li>
<li><a href="https://arxiv.org/abs/2410.16198"><b>Improve vision language model chain-of-thought reasoning</b></a></li>
<li><a href="https://arxiv.org/html/2501.12368v1"><b>InternLM-XComposer2. 5-Reward: A Simple Yet Effective Multi-Modal Reward Model</b></a></li>
<li><a href="https://arxiv.org/abs/2411.14432"><b>Insight-v: Exploring long-chain visual reasoning with multimodal large language models</b></a></li>
<li><a href="https://arxiv.org/abs/2503.12937"><b>R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy optimization</b></a></li>
<li><a href="https://arxiv.org/abs/2503.07065"><b>Boosting the generalization and reasoning of vision language models with curriculum reinforcement learning</b></a></li>
<li><a href="https://arxiv.org/abs/2412.14835"><b>Progressive multimodal reasoning via active retrieval</b></a></li>
<li><a href="https://arxiv.org/abs/2412.03548"><b>Perception tokens enhance visual reasoning in multimodal language models</b></a></li>
<li><a href="https://arxiv.org/abs/2404.03622"><b>Mind's Eye of LLMs: Visualization-of-Thought Elicits Spatial Reasoning in Large Language Models</b></a></li>
<li><a href="https://proceedings.mlr.press/v260/jia25b.html"><b>DCoT: Dual Chain-of-Thought Prompting for Large Multimodal Models</b></a></li>
<li><a href="https://aclanthology.org/2024.acl-long.432/"><b>Chain-of-exemplar: enhancing distractor generation for multimodal educational question generation</b></a></li>
<li><a href="https://arxiv.org/abs/2403.12488"><b>Dettoolchain: A new prompting paradigm to unleash detection ability of mllm</b></a></li>
<li><a href="https://arxiv.org/abs/2404.16033"><b>Cantor: Inspiring multimodal chain-of-thought of mllm</b></a></li>
<li><a href="https://arxiv.org/abs/2305.16582"><b>Beyond chain-of-thought, effective graph-of-thought reasoning in language models</b></a></li>
<li><a href="https://arxiv.org/abs/2311.17076"><b>Compositional chain-of-thought prompting for large multimodal models</b></a></li>
<li><a href="https://arxiv.org/abs/2309.01093"><b>Cotdet: Affordance knowledge prompting for task driven object detection</b></a></li>
</ul>

<h5> Video Reasoning </h5>
<ul>
<li><a href="https://ieeexplore.ieee.org/document/10376648/"><b>Intentqa: Context-aware video intent reasoning</b></a></li>
<li><a href="https://arxiv.org/abs/2307.16368"><b>Antgpt: Can large language models help long-term action anticipation from videos?</b></a></li>
<li><a href="https://arxiv.org/abs/2305.13903"><b>Let's think frame by frame with vip: A video infilling and prediction dataset for evaluating video chain-of-thought</b></a></li>
<li><a href="https://arxiv.org/abs/2501.03230"><b>Video-of-thought: Step-by-step video reasoning from perception to cognition</b></a></li>
<li><a href="https://arxiv.org/abs/2406.11333"><b>Hallucination mitigation prompts long-term video understanding</b></a></li>
<li><a href="https://arxiv.org/abs/2408.12009"><b>Cardiff: Video salient object ranking chain of thought reasoning for saliency prediction with diffusion</b></a></li>
<li><a href="https://arxiv.org/abs/2411.02570"><b>TI-PREGO: Chain of Thought and In-Context Learning for Online Mistake Detection in PRocedural EGOcentric Videos</b></a></li>
<li><a href="https://dl.acm.org/doi/10.1145/3696410.3714559"><b>Following clues, approaching the truth: Explainable micro-video rumor detection via chain-of-thought reasoning</b></a></li>
<li><a href="https://arxiv.org/abs/2502.06428"><b>CoS: Chain-of-Shot Prompting for Long Video Understanding</b></a></li>
<li><a href="https://arxiv.org/abs/2503.21776"><b>Video-r1: Reinforcing video reasoning in mllms</b></a></li>
<li><a href="https://arxiv.org/abs/2505.12434"><b>VideoRFT: Incentivizing Video Reasoning Capability in MLLMs via Reinforced Fine-Tuning</b></a></li>
<li><a href="https://arxiv.org/abs/2505.03318"><b>Unified multimodal chain-of-thought reward model through reinforcement fine-tuning</b></a></li>
<li><a href="https://arxiv.org/abs/2503.13444"><b>VideoMind: A Chain-of-LoRA Agent for Long Video Reasoning</b></a></li>
</ul>


<h5> Audio Reasoning </h5>
<ul>
<li><a href="https://arxiv.org/abs/2304.12995"><b>Audiogpt: Understanding and generating speech, music, sound, and talking head</b></a></li>
<li><a href="https://arxiv.org/html/2409.19510v1"><b>CoT-ST: Enhancing LLM-based Speech Translation with Multimodal Chain-of-Thought</b></a></li>
<li><a href="https://arxiv.org/abs/2501.10937"><b>Leveraging Chain of Thought towards Empathetic Spoken Dialogue without Corresponding Question-Answering Data</b></a></li>
<li><a href="https://arxiv.org/abs/2410.10676"><b>Both Ears Wide Open: Towards Language-Driven Spatial Audio Generation</b></a></li>
<li><a href="https://arxiv.org/abs/2503.02318"><b>Audio-reasoner: Improving reasoning capability in large audio language models</b></a></li>
<li><a href="https://arxiv.org/abs/2503.08540"><b>Mellow: a small audio language model for reasoning</b></a></li>
<li><a href="https://arxiv.org/abs/2503.11197"><b>Reinforcement learning outperforms supervised fine-tuning: A case study on audio question answering</b></a></li>
</ul>



<h4 id="mh"> 3. Multimodal Hallucination </h4>
<h5> Hallucination Cause Analysis </h5>
<ul>
<li><a href=""><b></b></a></li>
<li><a href=""><b></b></a></li>
<li><a href=""><b></b></a></li>
</ul>

<h5> Hallucination Detection </h5>
<ul>
<li><a href=""><b></b></a></li>
<li><a href=""><b></b></a></li>
<li><a href=""><b></b></a></li>
</ul>


<h5> Hallucination Mitigation </h5>
<ul>
<li><a href=""><b></b></a></li>
<li><a href=""><b></b></a></li>
<li><a href=""><b></b></a></li>
</ul>

<h4 id="ms"> 4. Multimodal Safety </h4>

<h4 id="ma"> 5. Multimodal Agents </h4>

<h4 id="vp"> 6. Video Perception </h4>



<h3 id="evaluation" align="center"> 🍒Evaluation </h3>

<h4 id="ke"> 1. Keyword Extraction </h4>

<h4 id="ee"> 2. Embedding-based Evaluation </h4>

<h4 id="me"> 3. MLLM-based Evaluation </h4>

<h4 id="ep"> 4. Evaluation Platform </h4>


<h3 id="rs" align="center"> 🥭Related Surveys </h3>

<h2 align="center"> 📚 Citation 📚</h2>
