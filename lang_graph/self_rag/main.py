from dotenv import load_dotenv
load_dotenv()

from graph.graph import app


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print(app.invoke({'question': 'Qual o municipio no circulo vicioso?'}))
    print(app.invoke({'question': 'Como fazer um pão?'}))


