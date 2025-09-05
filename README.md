# Chess AI
Simple neural network that gets the chess board as an input and return a 12bits vector that represent the next move.  
Total params: ``60,322,828``  
Trainned on an NVIDIA GeForce GTX 970 with an Intel Core i7 7700.  
**Output format :** (x,y)(x,y) => chess move | 12 dimensionnal vector (3bit per value)  
All the trainning data came from [pgnmentor](https://www.pgnmentor.com/files.html)  
## How to test it?  
**First, Clone the repo :**  
```
git clone https://github.com/abgache/chess-ai.git
cd chess-ai
```
**Then, download the requirements and run the main script:**  
```
pip install -r requirements.txt
python main.py
```
