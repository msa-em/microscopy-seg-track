---
title: 'Deep Learning Applications in Microscopy: Segmentation and Tracking'
short_title: Microscopy Segment Track
numbering:
  heading_2: false
---

+++ {"part": "abstract"} 

This paper reports the application of deep learning techniques in bright-field transmission electron microscopy image segmentation and tracking, in order to understand dynamic changes of nanoparticles during high-temperature sintering. Four state-of-the-art deep learning models, YOLOv8n-seg, Swin- UNet, VMamba, and EfficientSAM-tiny, were used and their performances were compared. The results show that EfficientSAM-tiny performs best in the segmentation task, achieving the highest accuracy (IoU 0.99672, Dice Coefficient 0.99836). In the tracking task, combining EfficientSAM-tiny with the DeAOT model successfully achieves efficient tracking and accurate identification of nanoparticles in microscope videos. The effectiveness and reliability of the method is verified by analyzing the video of the nanoparticle sintering process at a high temperature of 800°C. These results demonstrate the potential of deep learning techniques in microscope image analysis and introduce a new method from computer vision for microscope video tracking and analysis in materials science.
+++


+++{"part":"epigraph"}
:::{admonition} Co-First Authors
The authors marked with `#` contributed equally to this work as co-first authors.
:::
+++


+++{"part":"epigraph"}
:::{warning} Pre-print
This article has not yet been peer-reviewed.  
_Updated 2024 September 24_
:::
+++

+++ {"part": "acknowledgements"} 
_This work was carried out in part at the Singh Center for Nanotechnology, which is supported by the NSF National Nanotechnology Coordinated Infrastructure Program under grant NNCI-2025608 and through the use of facilities supported by the University of Pennsylvania Materials Research Science and Engineering Center (MRSEC) DMR-2309043. C.-Y. C. and E.A.S. acknowledge additional support through the NSF Division of Materials Research's Metals and Metallic Nanostructures program, DMR-2303084.Add your acknowledgments, if any, here._

**Declaration of Generative AI in Scientific Writing:**

_AI tools (GPT-4o) were used to improve the grammar and readability of this manuscript._
+++


+++ {"part": "competing interests"} 
## Competing Interests

The authors declare that they have no known competing personal relationships or financial interests that could have appeared to influence the work in this paper.
+++
