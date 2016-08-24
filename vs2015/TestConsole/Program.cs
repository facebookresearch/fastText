using FastText;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace TestConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            //Ensures that we are consistent against culture-specific number formating, etc...
            CultureInfo culture = CultureInfo.CreateSpecificCulture("en-US");

            CultureInfo.DefaultThreadCurrentCulture = culture;
            CultureInfo.DefaultThreadCurrentUICulture = culture;

            Thread.CurrentThread.CurrentCulture = culture;
            Thread.CurrentThread.CurrentUICulture = culture;


            if(!File.Exists(@"cheeseDisease.bin"))
            {
                Console.WriteLine("Please train the model first using the console version of fastText and the data supplied in the SampleData folder");
                Console.WriteLine("fastText.exe supervised -input cheeseDisease.txt -output cheeseDisease");
                return;
            }

            Console.WriteLine("Loading model, please wait");
            bool res = fastText.InitializeWithModel(@"cheeseDisease.bin");
            if (!res)
            {
                Console.WriteLine("Failed to load the model, might be damaged or you run out of memory!");
                return;
            }
            Console.WriteLine("... done!");

            var tests = GetTestData();
            int correct = 0, fail = 0, noLabel = 0;

            foreach(var test in tests)
            {
                var label = fastText.GetPrediction(test.Text, 1).Replace("__label__","").Replace("__","");
                if(label == "n/a")
                {
                    label = fastText.GetPrediction(test.Text, 1).Replace("__label__", "").Replace("__", "");
                }
                Console.WriteLine($"{test.Text} -> P:{label} / C:{test.Label}");
                correct += (label == test.Label) ? 1 : 0;
                fail    += (label == test.Label) ? 0 : 1;
                noLabel += (label == "n/a") ? 1 : 0;
            }

            Console.WriteLine($"Summary: {correct} correctly labeled, {fail-noLabel} mislabed, {noLabel} no labels found");
            Console.WriteLine("Press any key to finish!");
            Console.Read();

            fastText.Release();
        }


        struct TestValue
        {
            public string Label;
            public string Text;
        }
        static List<TestValue> GetTestData()
        {
            var Data = new List<TestValue>();
            Action<string,string> AddTest = delegate (string l, string t) { Data.Add(new TestValue() { Label = l, Text = t }); };

            AddTest("Disease", "Psittacosis");
            AddTest("Disease", "Cushing Syndrome");
            AddTest("Disease", "Esotropia");
            AddTest("Disease", "Jaundice, Neonatal");
            AddTest("Disease", "Thymoma");
            AddTest("Cheese", "Caerphilly");
            AddTest("Disease", "Teratoma");
            AddTest("Disease", "Phantom Limb");
            AddTest("Disease", "Iron Overload");
            AddTest("Disease", "Spermatic Cord Torsion");
            AddTest("Disease", "Epistaxis (Nosebleed)");
            AddTest("Cheese", "Folded cheese with mint");
            AddTest("Cheese", "Maytag Blue");
            AddTest("Cheese", "Castelmagno");
            AddTest("Disease", "Monoclonal Gammopathy");
            AddTest("Disease", "NiikawaKuroki Syndrome");
            AddTest("Cheese", "Cendre d'Olivet");
            AddTest("Disease", "Pericarditis");
            AddTest("Disease", "Speech Disorders");
            AddTest("Cheese", "Maredsous");
            AddTest("Disease", "Thyroid Diseases");
            AddTest("Cheese", "Briquette de Brebis");
            AddTest("Cheese", "Banon");
            AddTest("Disease", "Taeniasis");
            AddTest("Disease", "Testicular Diseases");
            AddTest("Disease", "Carcinoma");
            AddTest("Disease", "Rubella");
            AddTest("Disease", "Hordeolum");
            AddTest("Cheese", "Basing");
            AddTest("Disease", "Apnea");
            AddTest("Disease", "Urea Cycle Disorders");
            AddTest("Disease", "Infertility (Female Genital Diseases ..)");
            AddTest("Cheese", "Sardo");
            AddTest("Disease", "Ivemark Syndrome");
            AddTest("Cheese", "Ambert");
            AddTest("Disease", "Color Vision Defects");
            AddTest("Disease", "Arrhythmia");
            AddTest("Disease", "Tangier Disease");
            AddTest("Disease", "Herpes Labialis (Fever Blisters)");
            AddTest("Cheese", "Cathelain");
            AddTest("Disease", "Genital Warts");
            AddTest("Disease", "Pleurisy");
            AddTest("Disease", "Narcissism");
            AddTest("Disease", "ArnoldChiari Malformation");
            AddTest("Cheese", "Crottin du Chavignol");
            AddTest("Cheese", "White Stilton");
            AddTest("Disease", "Retinopathy of Prematurity");
            AddTest("Cheese", "Monterey Jack Dry");
            AddTest("Disease", "Prostatic Diseases");
            AddTest("Disease", "Psoriasis");
            AddTest("Cheese", "Lincolnshire Poacher");
            AddTest("Disease", "Lactose Intolerance");
            AddTest("Disease", "Pregnancy, Ectopic");
            AddTest("Disease", "Sinusitis");
            AddTest("Cheese", "Vignotte");
            AddTest("Disease", "Typhoid Fever");
            AddTest("Disease", "Diseases of Marine Mammals");
            AddTest("Disease", "Rhabdoid Tumor");
            AddTest("Disease", "Angioneurotic Edema");
            AddTest("Disease", "Amnesia");
            AddTest("Cheese", "Trois Cornes De Vendee");
            AddTest("Cheese", "Roncal");
            AddTest("Disease", "Avitaminosis");
            AddTest("Cheese", "Sourire Lozerien");
            AddTest("Disease", "CREST");
            AddTest("Disease", "Skin Ulcer");
            AddTest("Cheese", "Pave du Berry");
            AddTest("Disease", "Pain");
            AddTest("Disease", "Lymphogranuloma Venereum");
            AddTest("Disease", "Aneurysm");
            AddTest("Disease", "Taste Disorders");
            AddTest("Cheese", "Schabzieger");
            AddTest("Disease", "KleineLevin Syndrome");
            AddTest("Disease", "Facial Hemiatrophy (ParryRomberg Disease)");
            AddTest("Cheese", "Ardrahan");
            AddTest("Cheese", "Meyer Vintage Gouda");
            AddTest("Cheese", "Autun");
            AddTest("Disease", "Myositis");
            AddTest("Disease", "Vaginal Diseases");
            AddTest("Cheese", "Edam");
            AddTest("Disease", "Wilms' Tumor");
            AddTest("Cheese", "Maribo");
            AddTest("Disease", "Blue Rubber Bleb Nevus Syndrome");
            AddTest("Cheese", "Red Leicester");
            AddTest("Disease", "Trophoblastic Neoplasms");
            AddTest("Disease", "Eczema");
            AddTest("Disease", "Whipple's Disease");
            AddTest("Disease", "Alexanders Disease");
            AddTest("Cheese", "Tomme de Chevre");
            AddTest("Disease", "Gas Gangrene");
            AddTest("Disease", "Purpura, SchoenleinHenoch");
            AddTest("Disease", "Scoliosis");
            AddTest("Disease", "Vomiting");
            AddTest("Cheese", "Gouda");
            AddTest("Disease", "Arthropod Diseases");
            AddTest("Disease", "Orbital Cellulitis");
            AddTest("Disease", "Acne Rosacea");
            AddTest("Disease", "Menstruation Disturbances");
            AddTest("Disease", "Turner Syndrome");
            AddTest("Cheese", "Stinking Bishop");
            AddTest("Disease", "Adenomatous Polyposis Coli");
            AddTest("Disease", "Encephalomyelitis");
            AddTest("Disease", "Factor XI Deficiency");
            AddTest("Disease", "Shprintzen Syndrome");
            AddTest("Cheese", "Palet de Babligny");
            AddTest("Disease", "Lentigo, Malignant");
            AddTest("Disease", "Bronchiolitis");
            AddTest("Cheese", "Coulommiers");
            AddTest("Cheese", "Queso Majorero");
            AddTest("Disease", "Lipodystrophy");
            AddTest("Cheese", "Grana");
            AddTest("Disease", "Frog Diseases");
            AddTest("Cheese", "Anneau du Vic-Bilh");
            AddTest("Disease", "Thalamic Diseases");
            AddTest("Disease", "Chest Pain");
            AddTest("Disease", "Ichthyosis");
            AddTest("Cheese", "Klosterkaese");
            AddTest("Cheese", "Oschtjepka");
            AddTest("Disease", "Bronchopulmonary Dysplasia");
            AddTest("Disease", "Fanconi Anemia");
            AddTest("Cheese", "Gruyere");
            AddTest("Disease", "Scleroderma, Systemic");
            AddTest("Disease", "Hypothermia");
            AddTest("Cheese", "Ulloa");
            AddTest("Cheese", "Gornyaltajski");
            AddTest("Disease", "Beal's Syndrome");
            AddTest("Cheese", "Turunmaa");
            AddTest("Cheese", "Hereford Hop");
            AddTest("Disease", "Adrenogenital Syndrome");
            AddTest("Cheese", "Friesla");
            AddTest("Cheese", "Quatre-Vents");
            AddTest("Cheese", "Coeur de Chevre");
            AddTest("Disease", "Sialorrhea");
            AddTest("Disease", "Q Fever");
            AddTest("Disease", "Arachnoiditis");
            AddTest("Cheese", "Monterey Jack");
            AddTest("Disease", "Encephalocele");
            AddTest("Cheese", "Prastost");
            AddTest("Disease", "Desmoid Tumor");
            AddTest("Disease", "Torticollis");
            AddTest("Disease", "Anemia, Hemolytic");
            AddTest("Disease", "Sever's Disease");
            AddTest("Disease", "Arsenic Poisoning");
            AddTest("Cheese", "Nantais");
            AddTest("Disease", "Spina Bifida");
            AddTest("Disease", "Zenker Diverticulum");
            AddTest("Disease", "Conjunctivitis");
            AddTest("Disease", "Muscle Spasticity");
            AddTest("Cheese", "Kernhem");
            AddTest("Cheese", "Caciotta");
            AddTest("Cheese", "Abertam");
            AddTest("Disease", "Histoplasmosis");
            AddTest("Cheese", "Idaho Goatster");
            AddTest("Cheese", "Pecorino in Walnut Leaves");
            AddTest("Cheese", "Tommes");
            AddTest("Disease", "Hallux Valgus");
            AddTest("Disease", "Musculoskeletal Abnormalities (Pediatr.)");
            AddTest("Disease", "ChurgStrauss Syndrome");
            AddTest("Cheese", "Vasterbottenost");
            AddTest("Disease", "Peripheral Vascular Diseases");
            AddTest("Disease", "Constipation");
            AddTest("Disease", "Pin Worms");
            AddTest("Disease", "Dermoid Cyst");
            AddTest("Disease", "Catscratch Disease");
            AddTest("Disease", "Kabuki MakeUp Syndrome");
            AddTest("Disease", "Inflammation");
            AddTest("Cheese", "Gastanberra");
            AddTest("Cheese", "Fynbo");
            AddTest("Cheese", "Crottin de Chavignol");
            AddTest("Disease", "Lymphadenitis");
            AddTest("Cheese", "Pont l'Eveque");
            AddTest("Cheese", "Molbo");
            AddTest("Cheese", "Caprice des Dieux");
            AddTest("Disease", "Arial Fibrillation");
            AddTest("Cheese", "Macconais");
            AddTest("Cheese", "Sottocenare al Tartufo");
            AddTest("Cheese", "Quartirolo Lombardo");
            AddTest("Disease", "Scurvy");
            AddTest("Disease", "Fetal Alcohol Syndrome");
            AddTest("Disease", "Leukodystrophy, Metachromatic");
            AddTest("Disease", "Optic Neuritis");
            AddTest("Disease", "Herpes Zoster (Shingles)");
            AddTest("Cheese", "Rigotte");
            AddTest("Disease", "Trachoma");
            AddTest("Disease", "Dwarfism");
            AddTest("Disease", "Chorioretinitis");
            AddTest("Cheese", "Pas de l'Escalette");
            AddTest("Disease", "Hypophosphatasia");
            AddTest("Disease", "Alpha 1Antitrypsin Deficiency");
            AddTest("Disease", "Staphylococcal Infections");
            AddTest("Disease", "Rhinoscleroma");
            AddTest("Disease", "Paronychia");
            AddTest("Disease", "Autistic Disorder");
            AddTest("Disease", "Leprosy");
            AddTest("Disease", "Angina Pectoris");
            AddTest("Cheese", "Boulette d'Avesnes");

            return Data;
        }
    }
}
