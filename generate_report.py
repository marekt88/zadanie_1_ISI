from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY

def generate_report():
    doc = SimpleDocTemplate("technical_report.pdf", pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    Story = []
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))

    title = 'Technicka sprava: Regresny model pre predikciu'
    Story.append(Paragraph(title, styles['Title']))
    Story.append(Spacer(1, 12))

    text = '''
    1. Teoreticky opis pouziteho regresneho modelu

    V tomto projekte sme pouzili linearnu regresiu ako nas regresny model. Linearna regresia je statisticka metoda, 
    ktora sa pouziva na modelovanie linearneho vztahu medzi zavislou premennou a jednou alebo viacerymi nezavislymi 
    premennymi. Model predpoklada, ze vztah medzi premennymi moze byt vyjadreny linearnou funkciou.

    Matematicky mozeme linearnu regresiu vyjadrit ako:
    y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε

    kde y je zavisla premenna, x₁, x₂, ..., xₙ su nezavisle premenne, β₀ je intercept (hodnota y, ked vsetky x su 0), 
    β₁, β₂, ..., βₙ su koeficienty (vahy) pre kazdu nezavislu premennu, a ε je chybovy clen.

    2. Metodologia riesenia

    Pre implementaciu linearnej regresie sme zvolili kniznicu scikit-learn v Pythone. Tento vyber bol motivovany 
    niekolkymi faktormi:

    a) Jednoduchost pouzitia: scikit-learn poskytuje intuitivne API pre trenovanie a vyhodnocovanie modelov.
    b) Vykonnost: Implementacia v scikit-learn je optimalizovana a efektivna.
    c) Integracia: Lahko sa integruje s dalsimi nastrojmi pre manipulaciu s datami, ako numpy a pandas.

    Proces implementacie zahrnal nasledujuce kroky:
    1. Nacitanie dat pomocou numpy.
    2. Predspracovanie dat - konverzia kategorickych premennych pomocou LabelEncoder.
    3. Rozdelenie dat na trenovaciu a evaluacnu mnozinu.
    4. Trenovanie modelu linearnej regresie na trenovacich datach.
    5. Vyhodnotenie modelu pomocou R² skore.
    6. Predikcia na evaluacnych datach a ulozenie vysledkov.

    3. Diskusia o vysledkoch

    Nas model dosiahol R² skore 0.8236 na trenovacich datach, co naznacuje, ze model vysvetluje priblizne 82.36% 
    variability v cielovej premennej. Toto je pomerne dobry vysledok, ktory naznacuje, ze model zachytil vyznamne 
    vzory v datach.

    Avsak je dolezite poznamena, ze toto skore bolo dosiahnute na trenovacich datach, co moze viest k nadhodnoteniu 
    skutocnej vykonnosti modelu. Pre robustnejsie hodnotenie by bolo vhodne pouzit krizovu validaciu alebo samostatnu 
    validacnu mnozinu.

    Dalsie pristupy, ktore by mohli byt vyskusane pre potencialne zlepsenie vykonu modelu, zahrnaju:
    1. Pouzitie pokrocilejsich regresnych technik, ako je Ridge alebo Lasso regresia, ktore mozu pomoct s regularizaciou.
    2. Experimentovanie s roznymi metodami predspracovania dat, napriklad skalovanim funkcii.
    3. Skumanie nelinearnych vztahov v datach a potencialne pouzitie polynomialnych funkcii.

    Zaver:
    Nas linearny regresny model dosiahol slubne vysledky na trenovacich datach. Avsak pre komplexnejsie hodnotenie 
    a potencialne zlepsenie vykonu by bolo vhodne implementovat dalsie techniky a vykonat dokladnejsiu analyzu na 
    validacnej mnozine dat.
    '''

    for paragraph in text.split('\n\n'):
        Story.append(Paragraph(paragraph, styles['Justify']))
        Story.append(Spacer(1, 12))

    doc.build(Story)

if __name__ == '__main__':
    generate_report()
    print("Technical report generated: technical_report.pdf")
