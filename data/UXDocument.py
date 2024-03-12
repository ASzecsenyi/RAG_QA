from data import Document

chapters = [
    "Be Curious, Be Critical",
    "Developing for UX",
    "Evaluation Analysis",
    "Gathering User Requirements",
    "Human-in-the-Loop Systems and Digital Phenotyping",
    "In Real Life",
    "Modelling Requirements",
    "People are Complicated",
    "Practical Ethics",
    "Principles of Affective Experience (Emotion)",
    "Principles of Effective Experience (Accessibility)",
    "Principles of Efficient Experience (Usability)",
    "Principles of Engagement (Digital Umami)",
    "Prototyping and Rapid Application Development",
    "User Evaluation",
    "UXD and Visual Design",
    "What is UX",
]


class UXDocument(Document):
    """
    A document from the UX from 30,000ft dataset. Author: Simon Harper
    """

    def __init__(self, name: str = None, chapter: str = None):
        """
        :param name: the document identifier
        :type name: str, optional
        :param chapter: the id of the story to get data from
        :type chapter: str
        """

        if chapter is None:
            chapter = chapters[-1]

        if name is None:
            name = f"ux_{chapter}"

        # get chapter and questions string from txt file. It was copy pasted from website, we do not want escapes like \u2019 so correct encoding must be used
        with open(f"../data/files/ux_chapters/{chapter}.txt", "r", encoding="utf-8"
                  ) as f:
            file = f.read()
        document, questions = file.split("""Self Assessment Questions
Try these without reference to the text:""")

        # get questions
        questions = questions.split("\n")

        # remove empty questions
        questions = [question for question in questions if question != ""]

        questions = [
            {
                "question": question,
                "ground_truths": [""],
            }
            for question in questions
        ]

        super().__init__(document, name, questions)
