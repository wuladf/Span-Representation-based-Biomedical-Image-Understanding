# SpanPubMedBert based Biomedical Image Understanding
### Motivations
We want to understand the text embedded in biomedical figure so as to understand the figure.   
The text embedded in figure is hard to recognize due to the lack of unified figure standard, the different figure types(gel, bar, microscope and etc.), as well as the low figure quality.   
And even we recognize the text correctly, we still don't know the meaning of the text without a context.   
So it's nessary (of cource insufficiency) to analyse the context of figure to understand the figure text.   
The figure context mainly consist of figure caption and figure's reference sentences in fulltext. So we have two tasks(ner and role):

1. recognize the biomedical named entity(gene, protein, molecule, cell and etc.) in figure caption;
2. classify the entity roles(assayed, intervention, reporter and etc.). For what is entity role and the detailed meaning of each entity role, you can read the supp material(4 pages).

### Methods
For both tasks, we implement SpanPubMedBert， which is a combination of SpanBert and PubMedBert. We first recognize the entities in figure caption and then classify the recognized entity roles, that is, pipelineing.  

we also implement a multitask-SpanPubMedBert(MtSpanPubMedBert) model, to alleviate the error accumulation in ner task. The code and result will upload in a few days.  

### Results
Purely pipeline:  

ner F1: 85.57,  

role loosely F1: 85.63, strict F1: 82.17  

strict means both span and ner are correct, loosely means span correct only.  

### To-Do
For more results and the result analysis，read my paper(link later).  

I will write this ReadMe in more detail in later days.
