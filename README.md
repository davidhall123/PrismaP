Thank you for the opportunity to share my experience with you! I'm excited to continue the application process.
As a Ph.D. in Biomedical Engineering with focused expertise in AI and machine learning, I've successfully transitioned from academic research to commercial data science. My background combines rigorous research methodology from processing complex neural signals with practical experience building and deploying AI solutions. In my current role, I've independently developed an AI-powered SaaS platform from the ground up, handling everything from model architecture to production deployment. During my Ph.D., I developed custom imaging processing pipelines using scikit-learn and NumPy for neural signal analysis, which gave me a strong foundation in handling complex time series data and statistical analysis. This unique combination of deep technical knowledge and practical implementation experience positions me well to contribute to PrismaP's innovative work.
These examples are from my latest project working with Kemah Bay Marketing LLC and represent early versions of what are now in production models and databases. I was cleared my employer to share these as they are not currently being used in their current forms. 
Deep Learning and Transformers
Proficiency Level: Intermediate 
Experience: I developed and deployed a custom transformer model by fine-tuning LLava (from Hugging Face) to analyze book covers and summaries for genre prediction. This multilabel classification project handled 67 distinct genres and required building a robust data processing pipeline with automatic image augmentation to address class imbalance. I implemented Pytorch dataset/dataloader classes for efficient data handling, and used gradient accumulation with mixed precision training to optimize compute resources. The model achieved 67% accuracy on primary genre prediction and 84% accuracy on secondary genre identification.
GitHub: https://github.com/davidhall123/PrismaP/blob/main/FineTunedLlavaModel.py 

NLP and Transformers
Proficiency Level: Intermediate to Advanced
Experience: I built a multimodal book cover quality prediction system that integrates DeBERTa for text processing with image and genre features. The model analyzes normalized reviews, ratings, review velocity, cover images, summaries, and titles to evaluate cover quality. This project achieved 85% test accuracy, with 98% of predictions falling within one tier of the actual rating.
GitHub: https://github.com/davidhall123/PrismaP/blob/main/MultiModalBookTierClassifier.py

SQL and Database Management
Proficiency Level: Advanced
Experience: At Kemah Bay Marketing, I architected and implemented the entire database infrastructure for both data collection and production systems. I chose MySQL for its open-source nature and broad adoption, designing a hybrid system that uses cloud storage buckets for images while maintaining text and numeric data in MySQL. The system uses ISBN13 as primary keys to maintain data relationships. While I've built more complex database systems for the company, this is the implementation I'm cleared to share.
GitHub: https://github.com/davidhall123/PrismaP/blob/main/TrainingDatabase.py 

PyTorch
Proficiency Level: Intermediate to Advanced
Experience: I customized a Vision Transformer (ViT) for multi-label genre classification using book cover images. The implementation involved replacing the default classifier with a custom nn.Sequential module and integrating nn.Sigmoid() for probabilistic genre predictions. I optimized the training process using gradient accumulation to maximize performance on local hardware (RTX 4090), implementing comprehensive evaluation metrics including F1 scores and loss tracking.
GitHub: https://github.com/davidhall123/PrismaP/blob/main/VITGenreClassifier.py

Cloud Computing (Google Cloud)
Proficiency Level: Advanced
Experience: I've deployed multiple AI/ML models to Google Cloud using containerized microservices. My latest project, covertech.ai, focused on building secure, efficient backend services with comprehensive API coverage and robust failsafes. While my primary experience is with Google Cloud, I'm confident my background in complex backend architecture would transfer readily to AWS.
Live Implementation: https://covertech.ai/

LLM Implementation
Proficiency Level: Beginner-Intermediate    
Experience: I'm currently developing an open-source digital scribe system for healthcare providers using aaditya/OpenBioLLM-Llama3-8B and OpenAI's Whisper. The project aims to provide a free alternative to expensive EHR-integrated solutions, with FHIR compatibility for EPIC integration. So far, I've successfully implemented audio transcription through Whisper and SOAP note generation via OpenBioLLM, with direct JSON output in FHIR format. The project is in active development with friends- I'm currently working through some interesting challenges in the integration phase!
