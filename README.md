<div align="center">
    <h1>Neural Networks course - PWR Winter Semester 2023</h1> 
</div>

<div>
<h2>
1. Introduction:
</h2>
The human voice holds a vast amount of information beyond its role in communication. Through subtle nuances and patterns, valuable insights into an individual's health can be uncovered. Our project aims to utilize the power of neural networks to unlock the untapped diagnostic potential within vocal data.
</div>
<div>
<h2>
2. Scientific objective:
</h2>
The primary scientific objective of our project is to develop and optimize a neural network-based system for the accurate classification of various pathologies through the analysis of vocal patterns. This involves the following key components:

1. Dataset Preprocessing:
  <ul>
    <li>Gather a diverse and comprehensive dataset encompassing a range of pathologies.</li>
    <li>Implement preprocessing techniques to extract relevant features from the vocal data, considering utility of spectrograms.</li>
  </ul>

 2. Neural Network Architecture Design:
  <ul>
    <li>Explore and experiment with different neural network architectures, such as convolutional neural networks (CNNs) to identify the most effective model for pathology classification based on voice analysis.</li>
    <li>Optimize hyperparameters, including learning rates, layer configurations, and activation functions, to enhance the model's accuracy and generalization capabilities.</li>
  </ul>

3. Training and Validation:
  <ul>
    <li>Train the neural network on the prepared dataset, employing rigorous cross-validation techniques to ensure robust model performance.</li>
    <li>Implement transfer learning strategies, if applicable, to leverage pre-trained models.</li>
  </ul>

4. Summary

<div>
<h2>3. Saarbrücken Dataset Description</h2>

  <p>The Saarbrücken dataset is a curated collection designed for the analysis and classification of various vocal pathologies [Barry, W.J., Pützer, M.: Saarbrücken Voice Database, Institute of Phonetics, Univ. of Saarland, http://www.stimmdatenbank.coli.uni-saarland.de/]. </p>

  <h3>Pathological Categories and Distribution:</h3>

  <ul>
    <li><strong>Dysphonia:</strong> 101 samples</li>
    <li><strong>Functional Dysphonia:</strong> 112 samples</li>
    <li><strong>Hyperfunktionalle Dysphonia:</strong> 212 samples</li>
    <li><strong>Laryngitis:</strong> 140 samples</li>
    <li><strong>Recurrent Laryngeal Nerve Paresis:</strong> 213 samples</li>
  </ul>
    For this study we have chosen 5 most common pathologies.

  <h3>Healthy Samples:</h3>

  <p>In addition to the pathological recordings, the dataset includes 657 samples from healthy individuals.</p>

  <h3>Subdivision by Speech Elements:</h3>

  <ul>
    <li><strong>Vowels:</strong> The dataset includes recordings focusing on the stable articulation of the vowels /a/, /i/, and /e/ enabling a detailed examination of vowel-specific characteristics.</li>
    <li><strong>Utterance:</strong> A set of recordings captures the utterance of the phrase "Guten Morgen, wie geht's es Ihnen?" (Good morning, how are you?), offering insights into the impact of different pathologies on the pronunciation of common phrases.</li>
  </ul>
</div>

<div>
    <h2>4. Visual Representation</h2>
    Visualizing the intricate patterns and relationships within vocal data is crucial for understanding the effectiveness of our neural network-based pathology classification system. The following image provides a snapshot of the spectrogram analysis.
    
<div align="center">
    <img src="Images/spec_comparison.jpg" alt="NSpec" style="width: 800px; height: auto; vertical-align: middle; display: inline-block;" />
</div>
</div>


