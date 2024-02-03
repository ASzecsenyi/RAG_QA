# RAG_QA - Research Repository

This repository is dedicated to the RAG_QA project, which focuses on Retrieval Augmented Question Answering. The project aims to benchmark different retrieval strategies, employ various language models for answer generation, and evaluate performance using different datasets. 

## Project Structure

- `rag_qa/`: Django project root.
- `retrieval/`: Retrieval strategies.
- `qa/`: Question answering models.
- `data/`: Datasets for evaluation.
- `experiments/`: Experiment configurations, results, and logs.
- `notebooks/`: Jupyter notebooks for data analysis.
- `scripts/`: Standalone Python scripts.
- `doc/`: Research papers and documentation.
- `requirements.txt`: Required Python packages.
- `manage.py`: Django management script.

## Getting Started

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop)
2. Install [Anaconda](https://www.anaconda.com/products/individual)
3. Clone this repository
4. Open a terminal
5. Navigate to the root directory of the project:
    
    ```bash
    cd /path/to/RAG_QA
    ```
   
6. Create a virtual environment:

    ```bash
    conda create -n RAG_QA python=3.8
    ```
   
7. Activate the virtual environment:

    ```bash
    conda activate RAG_QA
    ```
   
8. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```
   
9. Start docker desktop
   
10. Run the following command in the root directory of the project:

    ```bash
    docker-compose build
    ```

## Usage

To run the interactive Django shell:

1. Do steps 1-10 in the 'Getting Started' section

2. Run the following command in the root directory of the project:

    ```bash
    docker-compose up
    ```

3. Navigate to http://localhost:8000/upload/ in your browser

4. Upload a text file, pose a question and click 'Upload'

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- TODO: Mention any contributors or sources of inspiration here.

