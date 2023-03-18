# Span Representation based Biomedical Image Understanding
### Motivations
We want to understand the text embedded in the biomedical figure to understand figure.   
The text embedded in the figure is hard to recognize due to the lack of a unified figure standard, the different figure types(gel, bar, microscope, etc.), as well as the low figure quality.   
And even if we recognize the text correctly, we still don't know the meaning of the text without context.   
So it's necessary (of course insufficiency) to analyze the context of the figure to understand the figure text.   
The figure context mainly consists of the figure caption and the figure's reference sentences in full text. So we have two tasks(ner and role):

1. recognize the biomedical named entity(gene, protein, molecule, cell, etc.) in figure caption;
2. classify the entity roles(assayed, intervention, reporter, etc.). For what is entity role and the detailed meaning of each entity role, you can read the supplementary material(4 pages).

### Methods
For both tasks, we implement span representation based PubMedBERT. We first recognize the entities in figure caption and then classify the recognized entity roles.     

### Results
Purely pipeline:  

ner F1: 85.57,  

role loosely F1: 85.63, strict F1: 82.17  

strict means both span and ner are correct, loosely means only span is correct.  

### To-Do
For more results and the result analysis, read my paper(link later).  

I will write this ReadMe in more detail in later days.
