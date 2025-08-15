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
      - [Collection](#collection)
      - [Synthesis](#synthesis)
      - [Annotation](#annotation)
      - [Cleaning](#cleaning)
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

<h3 align="center" id="data"> Data </h3>

<h4 id="collection"> Data Collection </h4>
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

<h4 id="synthesis"> Data Synthesis </h4>
<li><a href=""><b></b></a></li>
<h4 id="annotation"> Data Annotation </h4>

<h4 id="cleaning"> Data Cleaning </h4>


<h3 id="tasks" align="center"> Tasks </h3>

<h4 id="mrag"> Multimodal Retrieval Augmented Generation </h4>

<h4 id="mr"> Multimodal Reasoning </h4>

<h4 id="mh"> Multimodal Hallucination </h4>

<h4 id="ms"> Multimodal Safety </h4>

<h4 id="ma"> Multimodal Agents </h4>

<h4 id="vp"> Video Perception </h4>



<h3 id="evaluation" align="center"> Evaluation </h3>

<h4 id="ke"> Keyword Extraction </h4>

<h4 id="ee"> Embedding-based Evaluation </h4>

<h4 id="me"> MLLM-based Evaluation </h4>

<h4 id="ep"> Evaluation Platform </h4>


<h3 id="rs" align="center"> Related Surveys </h3>

<h2 align="center"> 📚 Citation 📚</h2>
