# RAG_QA - Research Repository

This repository is dedicated to the RAG_QA project, which focuses on Retrieval Augmented Question Answering. The project aims to benchmark different retrieval strategies, employ various language models for answer generation, and evaluate performance using different datasets. 

For results produced with this project, please refer to the [drive repo](https://livemanchesterac-my.sharepoint.com/:f:/g/personal/andras_szecsenyi_student_manchester_ac_uk/EtoQxq_66apJvMeiJDtEroEB59kFMGsEV6FfvEjrkFYoIw?e=hSdsJx). This should be accessible by all University accounts.

For questions, please contact the authors of the project at [andras.szecsenyi@student.manchester.ac.uk](mailto:andras.szecsenyi@student.manchester.ac.uk)

## Project Structure

- `experiment_app/`: Django app root.
- `rgqa_project/`: Django project.
- `retrieval/`: Retrieval strategies.
- `qa/`: Question answering models.
- `data/`: Datasets for evaluation.
- `experiments/`: Experiment configurations, results, and logs.
- `notebooks/`: Jupyter notebooks for data analysis.
- `scripts/`: Standalone Python scripts.
- `media/`: Uploaded files.
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
   
9. Set the environment variables HUGGINGFACE_API_KEY and OPENAI_API_KEY:

    ```bash
    export HUGGINGFACE_API_KEY=your_huggingface_api_key
    export OPENAI_API_KEY=your_openai_api_key
    ```
   
10. Start docker desktop
   
11. Run the following command in the root directory of the project:

    ```bash
    docker-compose build
    ```
    
12. Run the following command in the root directory of the project:

    ```bash
    docker-compose run web python manage.py migrate
    ```
    
13. Run the following command in the root directory of the project:

    ```bash
    docker-compose run web python manage.py createsuperuser
    ```


## Usage

To run the interactive Django shell:

1. Do steps 1-13 in the 'Getting Started' section

2. Run the following command in the root directory of the project:

    ```bash
    docker-compose up
    ```

3. Navigate to http://localhost:8000/ in your browser

4. Upload a text file, pose a question and click 'Upload'

5. To access the Django admin interface, navigate to http://localhost:8000/admin/ in your browser and log in with the superuser credentials created in step 13 of the 'Getting Started' section


## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- Thx Riza and Maksim for their guidance and support.

