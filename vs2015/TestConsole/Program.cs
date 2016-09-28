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


    class WordAndSim
    {
        public string word;
        public double sim;
    }

    namespace FastTextNER
    { 
        using QuickGraph;
        using QuickGraph.Algorithms;
    public class RecognizedToken
    {
        public RecognizedToken(string type) { Type = type.Split('_').Last(); }
        public List<Token> Tokens = new List<Token>();
        public string Type;

        public RecognizedToken Add(Token token) { Tokens.Add(token); return this; }
    }
    public class Token
    {
        public Token(string value, string label, double intensity) { Value = value; Label = label; Intensity = intensity; }
        public string Value { get; set; }
        public string Label { get; set; }
        public double Intensity { get; set; }
    }
    public class SentenceEdge : Edge<Token>
    {
        public SentenceEdge(Token source, Token target, string source_type, string target_type, double probability) : base(source, target) { SourceType = source_type; TargetType = target_type; Probability = probability; }
        public double Probability { get; set; }
        public string SourceType        { get; set; }
        public string TargetType        { get; set; }

        public string ToString()
        {
            return SourceType + "[" + Probability.ToString("0.00") + "]" + TargetType;
        }
    }

    public class SentenceInterpretation
    {
        public List<RecognizedToken> Sentence = new List<RecognizedToken>();
        public double Probability { get; set; }

        public void Add(RecognizedToken curToken)
        {
            Sentence.Add(curToken);
        }
    }
        public class SentenceGraph : AdjacencyGraph<Token, SentenceEdge>
        {
            public SentenceGraph() : base(allowParallelEdges: true) { }

            public const string NullType = "O";
            public const char BeginTag = 'B';
            public const char MiddleTag = 'M';
            public const char EndTag = 'E';
            public const char SingleTag = 'S';

            public static bool IsTransitionAllowed(string currentLabel, string nextLabel)
            {
                char currentTag = currentLabel[0];
                char nextTag = nextLabel[0];

                string currentType = currentLabel.Substring(2);
                string nextType = nextLabel.Substring(2);

                bool typesMatch = currentType == NullType || nextType == NullType || currentType == nextType;

                switch (currentTag)
                {
                    case BeginTag: { return typesMatch && (nextTag == MiddleTag || nextTag == EndTag); }
                    case MiddleTag: { return typesMatch && (nextTag == MiddleTag || nextTag == EndTag); }
                    case EndTag: { return typesMatch && (nextTag == SingleTag || nextTag == BeginTag); }
                    case SingleTag: { return (nextTag == SingleTag || nextTag == BeginTag); }
                }
                throw new Exception("Invalid tag " + currentTag);
            }
            public List<SentenceInterpretation> GetAllPossibleSentenceInterpretations(Token source, Token destination, int pathCount = 5)
            {
                Func<SentenceEdge, double> edgeWeights = (e) =>
                {
                //Console.WriteLine(e.Source.Value + "[" + e.SourceType + "]" + "->" + e.Target.Value + "[" + e.TargetType + "]" + (1 - e.Probability).ToString("0.00"));
                return (1 - e.Probability);
                };// IMPROVE!
                var AllPaths = new List<SentenceInterpretation>();


                //var tryGetPath = this.ShortestPathsDijkstra(edgeWeights, source);
                //IEnumerable<SentenceEdge> result;
                //var tmp = tryGetPath(destination, out result);


                foreach (IEnumerable<SentenceEdge> path in AlgorithmExtensions.RankedShortestPathHoffmanPavley(this.ToBidirectionalGraph(), edgeWeights, source, destination, pathCount))
                //var path = result.ToList();
                {
                    RecognizedToken curToken = new RecognizedToken(path.First().SourceType).Add(path.First().Source);

                    var currentInterpretation = new SentenceInterpretation();

                    bool fullCapture = false;
                    double probability = 1;

                    foreach (var edge in path)
                    {
                        probability *= edge.Probability;

                        switch (edge.SourceType[0])
                        {
                            case BeginTag: { curToken = new RecognizedToken(edge.SourceType).Add(edge.Source); fullCapture = false; break; }
                            case MiddleTag: { curToken.Add(edge.Source); fullCapture = false; break; }
                            case EndTag: { curToken.Add(edge.Source); fullCapture = true; break; }
                            case SingleTag: { curToken = new RecognizedToken(edge.SourceType).Add(edge.Source); fullCapture = true; break; }
                        }

                        if (fullCapture && edge.Source != source) { currentInterpretation.Add(curToken); curToken = null; }
                    }

                    var lastEdge = path.Last();
                    probability *= lastEdge.Probability;

                    switch (lastEdge.TargetType[0])
                    {
                        case BeginTag: { curToken = new RecognizedToken(lastEdge.TargetType).Add(lastEdge.Target); fullCapture = false; break; }
                        case MiddleTag: { curToken.Add(lastEdge.Target); fullCapture = false; break; }
                        case EndTag: { curToken.Add(lastEdge.Target); fullCapture = true; break; }
                        case SingleTag: { curToken = new RecognizedToken(lastEdge.TargetType).Add(lastEdge.Target); fullCapture = true; break; }
                    }
                    if (fullCapture && lastEdge.Target != destination) { currentInterpretation.Add(curToken); curToken = null; }

                    currentInterpretation.Probability = probability;

                    AllPaths.Add(currentInterpretation);
                }

                return AllPaths;
            }

            public static List<SentenceInterpretation> IdentifyEntities(fastText fastTextModel, string text, int numberOfPossibilities)
            {
                var paths = new List<SentenceInterpretation>();

                var words = text.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries).ToList(); //IMPROVE TOKENIZATION HERE

                if (words.Count == 0) { return paths; }

                var tokens = new List<List<Token>>();

                foreach (var w in words)
                {
                    var pred = fastTextModel.GetPrediction(w, 5);

                    if (pred.Count == 0)
                    {
                        pred.Add(new prediction() { label = "S_O", intensity = 1 });
                    }

                    tokens.Add(pred.Select(p => new Token(w, (p.label.Contains("_") ? p.label : "S_" + p.label), p.intensity)).ToList());
                }

                var graph = new SentenceGraph();

                tokens.ForEach(t => graph.AddVertexRange(t));

                //tokens.ForEach(p => Console.WriteLine("\t" + string.Join(" ", p.Select(t => t.Label + "[" + t.Intensity.ToString("0.0") + "]"))));


                for (int i = 0; i < (words.Count - 1); i++)
                {
                    double maxIntensityC = tokens[i].Max(p => p.Intensity);
                    double maxIntensityN = tokens[i + 1].Max(p => p.Intensity);
                    foreach (var source in tokens[i])
                    {
                        foreach (var dest in tokens[i + 1])
                        {
                            string clabel = source.Label;
                            string nlabel = dest.Label;
                            if (!source.Label.Contains("_")) { clabel = SentenceGraph.SingleTag + "_" + source.Label; }
                            if (!dest.Label.Contains("_")) { nlabel = SentenceGraph.SingleTag + "_" + dest.Label; }
                            if (SentenceGraph.IsTransitionAllowed(clabel, nlabel))
                            {
                                double probability = (source.Intensity / maxIntensityC) * (dest.Intensity / maxIntensityN);
                                if (source.Intensity < 0 || dest.Intensity < 0) { probability = 0; }

                                graph.AddEdge(new SentenceEdge(source, dest, clabel, nlabel, probability));
                                //Console.WriteLine($"\tFound {source.Value}[{clabel}] -> {dest.Value}[{nlabel}] with probablity {probability} and intensities {source.Intensity} and {dest.Intensity}");
                            }
                        }
                    }
                }

                var BoS = new Token("__BEGIN__", "", 1);
                var EoS = new Token("__END__", "", 1);
                graph.AddVertex(BoS); graph.AddVertex(EoS);

                foreach (var t in tokens.First())
                {
                    string tlabel = t.Label;
                    if (!t.Label.Contains("_")) { tlabel = SentenceGraph.SingleTag + "_" + t.Label; }
                    if (SentenceGraph.IsTransitionAllowed("S_O", tlabel))
                    {
                        graph.AddEdge(new SentenceEdge(BoS, t, "S_O", tlabel, 1.0));
                    }
                }

                foreach (var t in tokens.Last())
                {
                    string tlabel = t.Label;
                    if (!t.Label.Contains("_")) { tlabel = SentenceGraph.SingleTag + "_" + t.Label; }
                    if (SentenceGraph.IsTransitionAllowed(tlabel, "S_O"))
                    {
                        graph.AddEdge(new SentenceEdge(t, EoS, tlabel, "S_O", 1.0));
                    }
                }


                paths = graph.GetAllPossibleSentenceInterpretations(BoS, EoS, numberOfPossibilities);

                //paths.ForEach(p => Console.WriteLine("Probability: " + p.Probability.ToString("0.00") + "\t" + string.Join(" ", p.Sentence.Select(n => string.Join(" ", n.Tokens.Select(t => t.Value + "[" + t.Label + "]" ) ) ))));
                return paths;
            }
        }

    }

    class Program
    {

        static void Main3(string[] args)
        {
            //Ensures that we are consistent against culture-specific number formating, etc...
            CultureInfo culture = CultureInfo.CreateSpecificCulture("en-US");

            CultureInfo.DefaultThreadCurrentCulture = culture;
            CultureInfo.DefaultThreadCurrentUICulture = culture;

            Thread.CurrentThread.CurrentCulture = culture;
            Thread.CurrentThread.CurrentUICulture = culture;

            string model = @"C:\IW\VS\PROJECTS\BOSON\Boson.Testing.CreateLuisApp\bin\x64\Debug\training\ner-caseinsensitive-fasttext";
            var fastTextModel = new fastText(model + ".bin");

            //var tags = File.ReadAllLines(model + @".vec").Skip(2).Where(l => l.StartsWith("nertag")).Select(l => l.Split(' ').First()).ToList();

            while (true)
            {
                var text = Console.ReadLine();

                var paths = FastTextNER.SentenceGraph.IdentifyEntities(fastTextModel, text, 5);
                paths.ForEach(p => Console.WriteLine("Probability: " + p.Probability.ToString("0.00") + "\t" + string.Join(" ", p.Sentence.Select(n => string.Join(" ", n.Tokens.Select(t => t.Value + "[" + t.Label + "]" ) ) ))));


                //var pred = fastTextModel.GetPrediction(text, 5);
                //pred.ForEach(p => Console.WriteLine(p.label + "[" + p.intensity.ToString("0.00") + "]"));
                //foreach (var t in tags)
                //{
                //    Console.WriteLine(t + " [" + fastTextModel.GetWordSimilarity(text, t).ToString("0.00") + "]");
                //}
            }
            return;
        }

 


        static void Main(string[] args)
        { 


            string model = @"C:\BigData\NLPmodels\FastText\aviation-caseinsensitive";

            if(args.Length > 0) { model = args[0]; }

            //var words = File.ReadAllLines(model + @".vec").Skip(2).Select((l) => new WordAndSim() { word = l.Split(' ').First() }).ToList();
            //words.RemoveAll(x => x.word.Length < 6);


            var fastTextModel = new fastText(model + @".bin");

            var words = fastTextModel.GetWords();


            //var utterances = File.ReadAllLines(@"C:\stanford-nlp\classifier_training\test.tsv").Select(l => new IntentExample() { Intent = l.Split('\t').First(), Example = l.Split('\t').Last().ToLowerInvariant() }).ToList();


            while (true)
            {

                Console.Write("\nWord: "); var w1 = Console.ReadLine();

                Console.WriteLine("\nMost similar:");
                fastTextModel.GetMostSimilar(w1, 20).ForEach(ws => Console.WriteLine("\t" + ws.Item1.PadLeft(15) + " " + ws.Item2.ToString("0.00")));

                Console.WriteLine("\nLeast similar:");
                fastTextModel.GetLeastSimilar(w1, 5).ForEach(ws => Console.WriteLine("\t" + ws.Item1.PadLeft(15) + " " + ws.Item2.ToString("0.00")));



                //Console.WriteLine();

                //Console.Write("Parent: "); var parent = Console.ReadLine();

                //double[] averageDiff = new double[fastTextModel.GetVectorSize()];
                //double count = 0;
                //while(true)
                //{
                //    Console.Write("Child : "); var child = Console.ReadLine();

                //    if(string.IsNullOrWhiteSpace(child)) { break; }

                //    var tmpdiff= fastTextModel.GetWordDifference(child, parent);
                //    averageDiff = fastText.Add(averageDiff, tmpdiff);
                //}

                //fastText.Multiply(averageDiff, 1 / count);
                

                //Console.Write("New Parent : "); var newParent = Console.ReadLine();
                //Console.Write("New Child  : "); var newChild  = Console.ReadLine();

                ////var diff = fastTextModel.GetWordDifference(child, parent);
                //var newDiff = fastTextModel.GetWordDifference(newChild,newParent);

                //Console.WriteLine(string.Join("; ", averageDiff.Select(a => a.ToString("0.00"))));
                //Console.WriteLine(string.Join("; ", newDiff.Select(a => a.ToString("0.00"))));

                //Console.WriteLine("Similarity between diff vectors:" + fastText.CalculateCosineSimilarity(averageDiff, newDiff));

                


                //if(string.IsNullOrWhiteSpace(parent) || string.IsNullOrWhiteSpace(newParent)) { break; }


                //var newChildVector = fastText.Add(fastTextModel.GetWordVector(newParent), averageDiff);

                //Console.WriteLine("Similarity between projected new parent and new child: " + fastTextModel.GetWordSimilarity(newChild, newChildVector));


                //foreach (var w in words)
                //{
                //    w.sim = fastTextModel.GetWordSimilarity(w.word, newChildVector);
                //}

                //words.Sort((a, b) => b.sim.CompareTo(a.sim));


                //Console.WriteLine("Most similar:");
                //foreach (var w in words.Take(20))
                //{
                //    Console.WriteLine($"\t{w.word} -> {w.sim}");
                //}

                ////words.Reverse();

                ////Console.WriteLine();

                ////Console.WriteLine("Least similar:");
                ////foreach (var w in words.Take(5))
                ////{
                ////    Console.WriteLine($"\t{w.word} -> {w.sim}");
                ////}



                //Console.WriteLine();
            }

            fastText.Release();
            //while (true)
            //{
            //    Console.Write("First word: "); var w1 = Console.ReadLine();
            //    Console.Write("Second word: "); var w2 = Console.ReadLine();
            //    Console.WriteLine($"Similarity between {w1} and {w2} is {fastText.GetWordSimilarity(w1, w2)}");
            //    Console.WriteLine();
            //}
        }

        static void Main2(string[] args)
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
            var fastTextModel  = new fastText(@"cheeseDisease.bin");
 
            Console.WriteLine("... done!");

            var tests = GetTestData();
            int correct = 0, fail = 0, noLabel = 0;

            foreach(var test in tests)
            {
                var label = fastTextModel.GetPrediction(test.Text, 1).First().label.Replace("__label__","").Replace("__","");
                if(label == "n/a")
                {
                    label = fastTextModel.GetPrediction(test.Text, 1).First().label.Replace("__label__", "").Replace("__", "");
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

    class IntentExample
    {

        public string Example { get; set; }
        public string Intent { get; set; }
        public double Similarity { get; set; }
    }
}
