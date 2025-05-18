# LiveBench: A Challenging, CONTAMINATION-LIMITED LLM BENCHMARK

> LiveBench: A Challenging, CONTAMINATION-LIMITED LLM BENCHMARK
>
> LiveBench: 一个具有挑战性的、限制数据污染的大语言模型基准测试
>
> 作者是：Colin White。CMU的博士，然后现在在一个19年创的AI公司工作。
>
> 2024写的，发在了2025的ICLR上，还是挺新的工作的

[toc]

## 0. Abstract

Test set contamination, wherein test data from a benchmark ends up in a newer model's training set, is a well-documented obstacle for fair LLM evaluation and can quickly render benchmarks obsolete. To mitigate this, many recent benchmarks crowdsource new prompts and evaluations from human or LLM judges; however, these can introduce significant biases, and break down when scoring hard questions. In this work, we introduce a new benchmark for LLMs designed to be resistant to both test set contamination and the pitfalls of LLM judging and human crowdsourcing. We release LiveBench, the first benchmark that (1) contains frequently-updated questions from recent information sources, (2) scores answers automatically according to objective ground-truth values, and (3) contains a wide variety of challenging tasks, spanning math, coding, reasoning, language, instruction following, and data analysis. To achieve this, Li veBench contains questions that are based on recently-released math competitions, arXiv papers, news articles, and datasets, and it contains harder, contamination-limited versions of tasks from previous benchmarks such as Big-Bench Hard, AMPS, and IFEval. We evaluate many prominent closed-source models, as well as dozens of open-source models ranging from 0.5 B to 405 B in size. LiveBench is difficult, with top models achieving below $70 \%$ accuracy. We release all questions, code, and model answers. Questions are added and updated on a monthly basis, and we release new tasks and harder versions of tasks over time so that LiveBench can distinguish between the capabilities of LLMs as they improve in the future. We welcome community engagement and collaboration for expanding the benchmark tasks and models.

> 测试集污染，即基准测试中的测试数据出现在较新模型的训练集中，是公平评估 LLM 的一个众所周知的障碍，并且可能迅速使基准测试过时。为缓解这一问题，许多近期基准测试通过人类或 LLM 评审众包新的提示和评估；然而，这些方法可能引入显著偏差，并在评分难题时失效。在本工作中，我们引入了一个新的 LLM 基准测试，旨在抵抗测试集污染以及 LLM 评审和人类众包的陷阱。我们发布了 LiveBench，这是第一个（1）包含来自最新信息源的频繁更新问题，（2）根据客观的真实值自动评分答案，以及（3）包含涵盖数学、编码、推理、语言、指令执行和数据分析的多种挑战性任务的基准测试。为实现这一目标，LiveBench 包含基于最近发布的数学竞赛、arXiv 论文、新闻文章和数据集的问题，并包含来自先前基准测试如 Big-Bench Hard、AMPS 和 IFEval 的更难且污染受限的任务版本。 我们评估了许多知名的闭源模型，以及从 0.5B 到 405B 规模的数十个开源模型。LiveBench 具有较高难度，顶级模型的准确率低于 70%。我们发布所有问题、代码和模型答案。问题每月新增和更新，并且我们会随着时间推移发布新的任务和更难的任务版本，以便 LiveBench 能够区分 LLMs 未来能力的提升。我们欢迎社区参与和合作，共同扩展基准任务和模型。

## 1. Introduction

In recent years, as large language models (LLMs) have risen in prominence, it has become increasingly clear that traditional machine learning benchmark frameworks are no longer sufficient to evaluate new models. Benchmarks are typically published on the internet, and most modern LLMs include large swaths of the internet in their training data. If the LLM has seen the questions of a benchmark during training, its performance on that benchmark will be artificially inflated (referred to as "test set contamination") (Roberts et al., 2024; Dong et al., 2024; Deng et al., 2023; Golchin \& Surdeanu, 2023b), hence making many LLM benchmarks unreliable. Recent evidence of test set contamination includes the observation that LLMs' performance on Codeforces plummet after the training cutoff date of the LLM (Roberts et al., 2024; Jain et al., 2024), and before the cutoff date, performance is highly correlated with the number of times the problem appears on GitHub (Roberts et al., 2024). Similarly, a recent hand-crafted variant of the established math dataset, GSM8K, shows evidence that several models have overfit to this benchmark (Zhang et al., 2024; Cobbe et al., 2021).

To lessen dataset contamination, benchmarks using LLM or human prompting and judging have become increasingly popular (Jain et al., 2024; Chiang et al., 2024; Zheng et al., 2024; Li et al., 2024). However, using these techniques comes with significant downsides. While LLM judges have multiple advantages, such as their speed and ability to evaluate open-ended questions, they are prone to making mistakes and can have several biases (Li et al., 2024). Furthermore, LLMs often favor their own answers over other LLMs, and LLMs favor more verbose answers (Li et al., 2024; Dubois et al., 2024; Li et al., 2023b). Additionally, using humans to provide evaluations of LLMs can inject biases such as formatting of the output and the tone of the writing (Chiang et al., 2024). Using humans to generate questions also presents limitations. Human participants might not ask diverse questions, may favor certain topics that do not probe a model's general capabilities, or may construct their prompts poorly (Zheng et al., 2024).

In this work, we introduce a framework for benchmarking LLMs designed to minimize both test set contamination and the pitfalls of LLM judging and human crowdsourcing. We use this framework to create LiveBench, the first benchmark with these three desiderata: (1) LiveBench contains frequently-updated questions based on recent information sources; (2) LiveBench is scored automatically according to the objective ground truth without the use of an LLM judge; and (3) LiveBench questions are drawn from a diverse set of six categories. We ensure (2) by only including questions that have an objectively correct answer. LiveBench questions are difficult: no current model achieves higher than $70 \%$ accuracy. Questions are added and updated on a monthly basis, and we release new tasks and harder versions of tasks over time so that LiveBench can distinguish among the capabilities of LLMs as they improve in the future.

**Overview of tasks**. LiveBench currently consists of 18 tasks across 6 categories: math, coding, reasoning, language, instruction following, and data analysis. Each task falls into one of two types: (1) tasks which use an information source for their questions, e.g., data analysis questions based on recent Kaggle datasets, or fixing typos in recent arXiv abstracts; and (2) tasks which are more challenging or diverse versions of existing benchmark tasks, e.g., from Big-Bench Hard (Suzgun et al., 2023) or IFEval (Zhou et al., 2023a). The categories and tasks included in LiveBench are:

- **Math**: modified questions based on high school math competitions from the past 11 months, as well as harder versions of AMPS questions (Hendrycks et al., 2021)
- **Coding**: code generation questions from recent Leetcode and AtCoder questions (via LiveCodeBench (Jain et al., 2024)), as well as a novel code completion task
- **Reasoning**: a harder version of Web of Lies from Big-Bench Hard (Suzgun et al., 2023), and novel Zebra Puzzles (e.g., (Jeremy, 2009)) and spatial reasoning questions
- **Language Comprehension**: Connections word puzzles, a typo-fixing task, and a movie synopsis unscrambling task for recent movies on IMDb and Wikipedia
- **Instruction Following**: four tasks to paraphrase, simplify, summarize, or generate stories about recent new articles from The Guardian (Guardian Media Group, 1821), subject to one or more instructions such as word limits or incorporating specific elements in the response
- **Data Analysis**: three tasks using recent datasets from Kaggle and Socrata, specifically, table reformatting (among JSON, JSONL, Markdown, CSV, TSV, and HTML), predicting which columns can be used to join two tables, and predicting the correct column type annotation

We evaluate dozens of models, including proprietary models as well as open-source models with sizes ranging from 0.5 B to $8 \times 22 \mathrm{~B}$. We release all questions, code, and model answers, and we welcome community engagement and collaboration. Our codebase is available at https://github.com/livebench/livebench, and our leaderboard is available at https://livebench.ai.

## 2. Livebench Description

In this section, we introduce LiveBench. It currently has six categories: math, coding, reasoning, data analysis, instruction following, and language comprehension. Categories are diverse with two to four tasks per problem. Each task either includes recent information sources (such as very recent news articles, movie synopses, or datasets) or is a more challenging, more diverse version of an existing benchmark task.

Each task is designed to have $40-100$ questions which span a range of difficulty, from easy to very challenging, while loosely aiming for an overall $30-70 \%$ success rate on the top models for each task. Prompts are tailored for each category and task but typically include the following: zero-shot chain of thought (Kojima et al., 2022; Wei et al., 2022), asking the model to make its best guess if it does not know the answer, and asking the LLM to output its final answer in a way that is easy to parse, such as in XML tags or in **double asterisks**. We also acknowledge that parsing answers in this way requires some degree of instruction following, and we address this in Appendix A.4. In the following sections, we give a summary of each task from each category. See Appendix A. 3 for additional details.

### a. MATH CATEGORY

Evaluating the mathematical abilities of LLMs has been one of the cornerstones of recent research in LLMs, featuring prominently in many releases and reports (Reid et al., 2024; OpenAI, 2023; Bubeck et al., 2023). Our benchmark includes math questions of three types: questions modified from recent high school math competitions, fill-in-the-blank questions from recent olympiad competitions, and questions from our new, harder version of the AMPS dataset (Hendrycks et al., 2021).

Our first two math tasks, Competitions and Olympiad, are based on expert human-designed math problems that offer a wide variety in terms of problem type and solution technique. In Competitions, we include questions from AMC12 2023, SMC 2023, and AIME 2024 modifying the prose and the answer order; in Olympiad, we include questions based on USAMO 2024 and IMO 2024, in which the task is to rearrange masked out equations from the solution into the correct order. These questions test problem solving with algebra, combinatorics, geometry, logic, number theory, probability, and other secondary school math topics (Faires \& Wells, 2022).

Finally, we release synthetically generated math questions in the AMPS_Hard task. This task is inspired by the math question generation used to create the MATH and AMPS datasets (Hendrycks et al., 2021). We generate harder questions by drawing random primitives, using a larger and more challenging distribution than AMPS across 10 of the hardest tasks within AMPS.

### b. CODING CATEGORY

The coding ability of LLMs is one of the most widely studied and sought-after skills for LLMs (Mnih et al., 2015; Jain et al., 2024; Li et al., 2023a). We include two coding tasks in LiveBench: a modified version of the code generation task from LiveCodeBench (LCB) (Jain et al., 2024), and a novel code completion task combining LCB problems with partial solutions collected from GitHub.

The LCB Generation assesses a model's ability to parse a competition coding question statement and write a correct answer. We include 78 questions from LiveCodeBench (Jain et al., 2024) which has several tasks to assess the coding capabilities of large language models.

The Completion task specifically focuses on the ability of models to complete a partially correct solution-assessing whether a model can parse the question, identify the function of the existing code, and determine how to complete it. We use LeetCode easy, medium, and hard problems from LiveCodeBench's (Jain et al., 2024) April 2024 release, combined with matching solutions from https://github.com/kamyu104/LeetCode-Solutions, omitting the last 15-70\% of each solution and asking the LLM to complete the solution.

### c. REASONING CATEGORY

The reasoning ability of large language models is another highly benchmarked and analyzed skill of LLMs (Wei et al., 2022; Suzgun et al., 2023; Yao et al., 2024). In Li veBench, we include three reasoning tasks: our harder version of a task from Big-Bench Hard (Suzgun et al., 2023), Zebra puzzles, and spatial reasoning questions.

Web of Lies v2 is an advancement of the similarly named task included in Big-Bench (bench authors, 2023) and Big-Bench Hard (Suzgun et al., 2023). The task is to evaluate the truth value of a random Boolean function expressed as a natural-language word problem. We create new, significantly harder questions by including additional deductive components and several types of red herrings. Next, we include spatial reasoning questions. This set of 50 handwritten questions tests a model's ability to make deductions about intersections and orientations of common 2D and 3D shapes.

Finally, we include Zebra Puzzles, a well-known reasoning task (Jeremy, 2009) that tests the ability of the model to follow a set of statements that set up constraints, and then logically deduce the requested information. We build on an existing repository for procedural generation of Zebra puzzles (quint $t, 2023$ ). Below, we provide an example question from the Zebra Puzzles task.

```text
An example question from the Zebra Puzzle task.
here are 3 people standing in a line numbered 1 through 3 in a left to right order.
Each person has a set of attributes: Food, Nationality, Hobby.
The attributes have the following possible values:

- Food: nectarine, garlic, cucumber
- Nationality: chinese, japanese, thai
- Hobby: magic-tricks, filmmaking, puzzles
	and exactly one person in the line has a given value for an attribute.
	Given the following premises about the line of people:
- the person that likes garlic is on the far left
- the person who is thai is somewhere to the right of the person who likes magic-tricks
- the person who is chinese is somewhere between the person that likes cucumber and the person who likes puzzles
	Answer the following question: What is the hobby of the person who is thai? Return your answer as a single word, in the following format:**X**, where X is the answer.
```

### d. DATA ANALYSIS CATEGORY

LiveBench includes three practical tasks in which the LLM assists in data analysis or data science: column type annotation, table join prediction, and table reformatting. Each question makes use of a recent dataset from Kaggle or Socrata.

The first task is to predict the type of a column of a data table. To create a question for the column type annotation task (CTA), we randomly sample a table and randomly sample a column from that table. We use the actual name of that column as the ground truth and then retrieve some samples from that column. We provide the name of all the columns from that table and ask the LLM to select the true column name from those options.

Data analysts often also require a table to be reformatted from one type to another, e.g., from some flavor of JSON to CSV or from XML to TSV. We emulate that task in TableReformat by providing a table in one format and asking the LLM to reformat it into the target format.

Finally, another common application of LLMs in data analysis is performing table joins (Goldbloom, 2024; Liu et al., 2024b; Sheetrit et al., 2024). In the TableJoin task, the LLM is presented with two tables with partially overlapping sets of columns. The LLM is tasked with creating a valid join mapping from the first to the second table.

### e. INSTRUCTION FOLLOWING CATEGORY

An important ability of an LLM is its capability to follow instructions. To this end, we include instruction-following questions in our benchmark, inspired by IFEval (Zhou et al., 2023a), which is an instruction-following evaluation for LLMs containing verifiable instructions such as "write more than 300 words" or "Finish your response with this exact phrase: \{end_phrase\}." While IFEval used a list of 25 verifiable instructions, we use a subset of 16 that excludes instructions that do not reflect real-world use-cases. See Appendix Table 3. Furthermore, in contrast to IFEval, which presents only the task and instructions with a simple prompt like "write a travel blog about Japan", we provide the models with an article from The Guardian (Guardian Media Group, 1821), asking the models to adhere to multiple randomly-drawn instructions while asking the model to complete one of four tasks related to the article: Paraphrase, Simplify, Story Generation, and Summarize. We score tasks purely by their adherence to the instructions.

### f. LANGUAGE COMPREHENSION CATEGORY

Finally, we include multiple language comprehension tasks. These tasks assess the language model's ability to reason about language itself by, (1) completing word puzzles, (2) fixing misspellings while leaving other stylistic changes in place, and (3) reordering scrambled plots of unknown movies.

First, we include the Connections category. Connections is a word puzzle popularized by the New York Times (although similar ideas have existed previously). In this task, we present questions of varying levels of difficulty with 8, 12, and 16-word varieties. The objective of the game is to sort the words into sets of four words, such that each set has a 'connection' between them.

Next, we include the Typos task. The idea behind this task is inspired by the common use case for LLMs in which a user asks the LLM to identify typos and misspellings in some written text but to leave other aspects of the text unchanged. We create the questions for this task from recent ArXiv abstracts, which we ensure originally have no typos, by programmatically injecting common human typos into the text. Below is an example question from the Typos task.

```
An example question from the Typos task.

Please output this exact text, with no changes at all except for fixing the misspellings. Please leave all other stylistic decisions like commas and US vs British spellings as in the original text.

We inctroduce a Bayesian estimation approach forther passive localization of an accoustic source in shallow water using a single mobile receiver. The proposed probablistic focalization method estimates the timne-varying source location inther presense of measurement-origin uncertainty. In particular, probabilistic data assocation is performed to match tiome-differences-of-arival (TDOA) observations extracted from the acoustic signal to TDOA predicitons provded by the statistical modle. The performence of our approach is evaluated useing rela acoustic data recorded by a single mobile reciever.
```

Finally, we include the Plot Unscrambling task, which takes the plot synopses of recentlyreleased movies from IMDb or Wikipedia. We randomly shuffle the synopses sentences and then ask the LLM to simply reorder the sentences into the original plot. We find that this task is very challenging for LLMs, as it measures their abilities to reason through plausible sequences of events.

### g. LIVEBENCH UPDATES AND MAINTENANCE PLAN

Maintaining a contamination-limited benchmark requires that we update the set of questions over time. We have so far released two updates, and we plan to continue to release updates to add new questions and remove outdated questions. In each update, we replace $1 / 6$ of the questions on average, so that the benchmark is fully refreshed roughly every 6 months. We may speed up the turnover rate of questions in the future, based on interest in LiveBench. Each month, we do not release the new questions until one month later, so that the public leaderboard always has $1 / 6$ questions that are private. We choose tasks to update based primarily on two factors: (1) the oldest tasks, and (2) the currently easiest tasks. In this way, the questions in LiveBench will stay new and continue to challenge the most capable LLMs. See additional details, as well as a longer discussion on different forms of contamination, in Appendix A.6.

**Method for sustainability**  One downside in a frequently-updating benchmark is that it requires consistent work and computational resources each month. Therefore, we have a plan in place to ensure its continued success. We maintain the best (or most popular) $40-50$ models on the leaderboard so as to avoid an ever-growing list of models to evaluate each month. For example, we maintain about two versions of each model family on the leaderboard (to show the improvement from the most recent version) but no more. This ensures that we have a tractable set of at most 50 models to evaluate on 200 questions each month, which is easily within the computational budgets of the authors' institutions. Additionally, we have already had community contributions which further reduces the computational burden of the authors.

The only other recurring work is to update the questions themselves each month. While we are excited and able to add novel tasks each month, many of the tasks are synthetic and therefore very fast and simple to create a new set of questions based on fresh data (e.g., updating the typos task using brand new arXiv papers). Additionally, we have also seen community engagement here as well.

**Completed monthly updates**   In the first monthly update, we added 50 questions in a new spatial reasoning task, 28 additional coding generation questions, and 12 additional coding completion questions. The total size of the benchmark after this update became 1000. In the second monthly update, we fully updated the math olympiad questions, and we partially updated the math AMPS_Hard and math_comp questions, for 132 replaced questions, to maintain 1000 questions.

## 3. EXPERIMENTS

In this section, first we describe our experimental setup and present full results for 40 LLMs on all 18 tasks of LiveBench. Next, we give an empirical comparison of LiveBench to existing prominent LLM benchmarks, and finally, we present ablation studies.

Experimental setup. Our experiments include 40 LLMs total, with a mix of top proprietary models, large open-source models, and small open-source models. In particular, for proprietary models, we include OpenAI models such as o1-preview, chatgpt-4o, and gpt-4o (Brown et al., 2020; OpenAI, 2023), Anthropic models such as claude-3-5-sonnet-20240620, Google models such as gemini-1.5-pro-002 (Reid et al., 2024), and Mistral models such mistral-large-2407 (Jiang et al., 2023).

For open-source models, we include models such as Llama-3.1-405b-instruct, Llama-3.1-70b-instruct (Dubey et al., 2024), deepseek-v2.5, (Liu et al., 2024a), qwen2 .5-72b-instruct (Team, 2024b; Yang et al., 2024), command-r-plus-08-2024 (Cohere, 2024; Cohere For AI, 2024), gemma-2-27b-it (Team, 2024a; Team et al., 2024), mixtral-8x22b-instruct-v0.1 (Jiang et al., 2023), and phi-3.5-moe-instruct (Abdin et al., 2024). See Table 4 for a full list of citations.

For all models and tasks, we perform single-turn evaluation with temperature 0 , unless otherwise noted in the model card. All models run with their respective templates from our updated version of FastChat (Zheng et al., 2024). We run all open-source models with bfloat.16. When running new models, we take care to set up its hyperparameters and chat template as in the model's example code, and we also double check the outputs to make sure that the inference, as well as our automated parsing functions, are working correctly and fairly. See more details in Appendix A. 4 and Appendix A.5. For each question, a model receives a score from 0 to 1 . For each model, we compute the score on each task as the average of all questions, we compute the score on each of the six categories as the average of all their tasks, and we compute the final LiveBench score as the average of all six categories. In Appendix B, we give additional documentation including average input/output tokens and cost to run LiveBench for each API model.

### 3.1 DISCUSSION OF RESULTS

We compare all 40 models on LiveBench according to the experimental setup described above; see Table 1 and Table 2. We find that o1-preview-2024-09-12 performs the best overall, $6 \%$ better than all other models. o1-preview-2024-09-12 substantially outperforms all other models in the data analysis, language, and math categories. The next-best model is claude-3-5-sonnet-20240620, which far outperforms all other models in the coding category (although ol-mini outperforms claude-3.5 in code generation, claude-3.5 has the edge in code completion). o1-mini-2024-09-12 is third overall and is significantly better than all other models in the reasoning category.

The best-performing open-source models are 11ama-3.1-405b-instruct and qwen2.5-72b-inst ruct, which virtually tie with each other and outperform gpt-4-turbo. The best-performing small open-source model is phi-3.5-moe-inst ruct (see Table 2): with only 6.6 B active parameters, it outperforms $\mathrm{gpt}-3.5$ and is on par with mixt ral-8x22b .

### 3.2 CORRELATION ANALYSES

Now we present analyses involving correlation among different categories and tasks. First, we compute the Pearson correlation coefficient among all pairs of categories and tasks in LiveBench (see Figure 2). We find that unsurprisingly, math, coding, and reasoning all correlate with one another. Interestingly, language correlates fairly well with data analysis, likely due to both categories including tasks that require the LLM to output a large part of the prompt that is modified in a specific way (e.g., by fixing typos or changing the table format). Surprisingly, instruction following correlates relatively weakly with all other categories. Among tasks, we see that math comp correlates the highest with the average LiveBench performance, suggesting that this task is the greatest indicator of overall model performance. This is likely due to these being high-quality, diverse mathematical reasoning questions (which we modified to reduce contamination).

Next, in order to see the strengths and weaknesses of each model, we create a scatterplot of each model's overall LiveBench performance vs. performance on a single category or task (Figure 3). By plotting a best fit line and computing the residuals for each model, we can compute which models are outliers in specific categories - that is, models that are disproportionately stronger in a particular category relative to the best fit line. We see that the ol and phi series of models are outliers in terms of reasoning (Figure 3, left), while some of the Llama, gemini, and command-r models are outliers in terms of instruction following. We present additional details in Appendix A.1, including a table of each model's relative best and worst tasks (computed as the highest and lowest residuals).

### 3.3 COMPARISON TO OTHER LLM BENCHMARKS

Next, we compare LiveBench to two prominent benchmarks, ChatBot Arena (Chiang et al., 2024) and Arena-Hard (Li et al., 2024). In Figure 4, we show a bar plot comparison among models that are common to both benchmarks, and in Figure 6, we compare the performance of these models to a best-fit line. We also compute the correlation coefficient of model scores among the benchmarks: LiveBench has a 0.91 and 0.88 correlation with ChatBot Arena and Arena-Hard, respectively.
Based on the plots and the correlation coefficients, we see that there are generally similar trends to LiveBench, yet some models are noticeably stronger on one benchmark vs. the other. For example, gpt-4-0125-preview and gpt-4-turbo-2024-04-09 perform substantially better on Arena-Hard compared to LiveBench, likely due to the known bias from using gpt-4 itself as the LLM judge (Li et al., 2024). We hypothesize that the strong performance of some models such as the gemini-1.5 models on ChatBot Arena compared to LiveBench may be due to having an output style that is preferred by humans. These observations emphasize the benefit of using ground-truth judging, which is immune to biases based on the style of the output.

Comparison between Ground-Truth and LLM-Judging As an additional comparison between LiveBench and LLM judge based benchmarks, we give a preliminary study in the Appendix on the efficacy of LLM judging for hard math and reasoning questions. Specifically, we run an initial experiment regarding the question, 'if an LLM struggles to answer a hard math or reasoning question, then will the LLM also struggle to determine whether or not a given answer to that question is correct?' Our experiments give evidence that the answer is yes, for zebra puzzles and AMC/AIME questions, but the results are not definitive. See Appendix A.2.

### 3.4 ANALYSIS OF MONTHLY UPDATES

As described in Section 2.7, we have completed two monthly updates for LiveBench so far. The rank correlation between the original and first update, and the first and second update, are both $>0.997$, showing that the model rankings have stayed consistent. On the other hand, between the original and the most-recent set of questions, the median and mean average scores (among models included in all iterations of the leaderboard) have both dropped by about $1.2 \%$, showing that the benchmark is becoming harder over time, as newly released models become more capable.

## 4. CONCLUSIONS, LIMITATIONS, AND FUTURE WORK

In this work, we introduced LiveBench, an LLM benchmark designed to mitigate both test set contamination and the pitfalls of LLM judging and human crowdsourcing. LiveBench is the first benchmark that (1) contains frequently updated questions from new information sources, in which questions become harder over time, (2) scores answers automatically according to objective groundtruth values, without the use of LLM judges, and (3) contains a wide variety of challenging tasks, spanning math, coding, reasoning, language, instruction following, and data analysis. LiveBench contains questions that are based on recently released math competitions, arXiv papers, and datasets, and it contains harder, contamination-limited versions of previously released benchmarks. We released all questions, code, and model answers, and questions are added and updated on a monthly basis. We welcome community collaboration for expanding the benchmark tasks and models.

Limitations and Future Work. While we attempted to make LiveBench as diverse as possible, there are still additions from which it would benefit. For example, we hope to add non-English language tasks in the future. Furthermore, while ground truth scoring is beneficial in many ways, it still cannot be used for certain use cases, such as 'write a travel guide to Hawaii' in which it is hard to define a ground truth. Finally, while we attempted to make all tasks and categories fair for all models, there are still biases due to certain LLM families favoring certain prompt types. We plan to update the prompts (at the start and end of each question) in the future, as new prompt strategies are developed. Similarly, we plan to continue updating the LiveBench leaderboard as new LLMs are released.
