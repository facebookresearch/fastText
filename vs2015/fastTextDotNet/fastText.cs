/**
* Copyright (c) 2016-present, Rafael Fernandes de Oliveira - rafael@rafael.aero
* All rights reserved.
*
* This source code is licensed under the BSD-style license found in the
* LICENSE file in the root directory of this source tree. An additional grant
* of patent rights can be found in the PATENTS file in the same directory.
*/

using System;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.IO;
using System.Globalization;
using System.Threading;

namespace FastText
{

    public struct prediction
    {
        public string label;
        public double intensity;
    }

    public struct WordAndVector
    {
        public string   Word;
        public double[] Vector;
    }
    public class fastText
    {
        [DllImport("fastText.dll", EntryPoint = "initialize", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        private static extern void initialize();

        [DllImport("fastText.dll", EntryPoint = "dispose", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        private static extern void dispose();

        [DllImport("fastText.dll", EntryPoint = "loadModel", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I4)]
        private static extern int loadModel([MarshalAs(UnmanagedType.LPStr)] string filename);


        
        [DllImport("fastText.dll", EntryPoint = "getWordCount", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I4)]
        private static extern int getWordCount(int model_index);

        
        [DllImport("fastText.dll", EntryPoint = "getWord", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        private static extern void getWord(int model_index, int k, StringBuilder output);



        [DllImport("fastText.dll", EntryPoint = "predict", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        private static extern void predict(int model_index, StringBuilder text, int k);

        [DllImport("fastText.dll", EntryPoint = "getPredictionBufferSize", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I4)]
        private static extern int getPredictionBufferSize(int model_index);

        [DllImport("fastText.dll", EntryPoint = "getPrediction", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        private static extern void getPrediction(int model_index, StringBuilder output);

        [DllImport("fastText.dll", EntryPoint = "getVector", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        private static extern void getVector(int model_index, [MarshalAs(UnmanagedType.LPStr)] string word, double[] output);

        [DllImport("fastText.dll", EntryPoint = "getVectorSize", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I4)]
        private static extern int getVectorSize(int model_index);


        public static void InitializeFastText()
        {
            initialize();
        }



        public static void Release()
        {
            dispose();
        }

        public string labelStartMarker = "__label__";
        public string labelEndMarker = "__";

        public static List<string> Punctuation {get;set;} = new List<string>() { " ", ".", ",", ";", ":", "-", "/", @"\", "!", "?", "'", "\"", "[", "]", "{", "}", "(", ")", "<", ">", "$", "%", "&", "*", "+", "_", "#" };

        public static List<string> StopWords { get; set; } = new List<string>() { "a", "able", "about", "across", "after", "all", "almost", "also", "am", "among", "an", "and", "any", "are", "as", "at", "be", "because", "been", "but", "by", "can", "cannot", "could", "dear", "did", "do", "does", "either", "else", "ever", "every", "for", "from", "get", "got", "had", "has", "have", "he", "her", "hers", "him", "his", "how", "however", "i", "if", "in", "into", "is", "it", "its", "just", "least", "let", "like", "likely", "may", "me", "might", "most", "must", "my", "neither", "no", "nor", "not", "of", "off", "often", "on", "only", "or", "other", "our", "own", "rather", "said", "say", "says", "she", "should", "since", "so", "some", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "tis", "to", "too", "twas", "us", "wants", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would", "yet", "you", "your" };


        private int ModelIndex;

        private List<WordAndVector> WordsAndVectors { get; set; } = new List<WordAndVector>();

        public fastText(string filename)
        {
            InitializeFastText();
            ModelIndex = loadModel(filename);
        }

        
        public List<WordAndVector> GetWords()
        {
            int nwords = getWordCount(ModelIndex);
            if(WordsAndVectors.Count != nwords)
            {
                WordsAndVectors.Clear();
                var out_buffer = new StringBuilder(1000); //Must be enough :D
                for(int i = 0; i < nwords; i++)
                {
                    getWord(ModelIndex, i, out_buffer);
                    var wv    = new WordAndVector();
                    wv.Word   = out_buffer.ToString();
                    wv.Vector = GetWordVector(wv.Word);
                    WordsAndVectors.Add(wv);
                }
            }
            return WordsAndVectors;
        }

        public List<Tuple<string,double>> GetMostSimilar(string word, int count)
        {
            var wordVector = GetParagraphVector(word);
            return GetWords().Select(w => new Tuple<string,double>(w.Word, CalculateCosineSimilarity(wordVector, w.Vector)))
                             .OrderBy(t => -t.Item2)
                             .Take(count).ToList();
        }

        public List<Tuple<string,double>> GetLeastSimilar(string word, int count)
        {
            var wordVector = GetParagraphVector(word);
            return GetWords().Select(w => new Tuple<string,double>(w.Word, CalculateCosineSimilarity(wordVector, w.Vector)))
                             .OrderBy(t => t.Item2)
                             .Take(count).ToList();
        }


        public int GetVectorSize()
        {
            return getVectorSize(ModelIndex);
        }

        public List<prediction> GetPrediction(string text, int numberOfTopLabelsToReturn)
        {
            var predictions = new List<prediction>();

            var buffer = new StringBuilder(text + "\n");
            predict(ModelIndex, buffer, numberOfTopLabelsToReturn);
            var out_buffer = new StringBuilder(getPredictionBufferSize(ModelIndex));
            getPrediction(ModelIndex,out_buffer);

            string tmp = out_buffer.ToString().TrimEnd(new char[] { '\n', '\r' });
            if(tmp=="n/a") { return predictions; }

            var labels = tmp.Split(new char[] { ';' }, StringSplitOptions.RemoveEmptyEntries);
            
            foreach(var lbl in labels)
            {
                var parts = lbl.Replace("]","").Split('[').ToList();
                predictions.Add(new prediction() { label = parts.First().Replace(labelStartMarker,"").Replace(labelEndMarker,""), intensity = double.Parse(parts.Last())});
            }

            return predictions;
        }

        public double[] GetParagraphVector(string text)
        {
            var words = text.Split(Punctuation.ToArray(), StringSplitOptions.RemoveEmptyEntries).ToList();
            var vecSum = new double[getVectorSize(ModelIndex)];

            words = words.Except(StopWords).ToList();

            foreach (var word in words)
            {
                var vecWord = GetWordVector(word);
                vecSum = Add(vecSum, vecWord);
            }

            vecSum = Multiply(vecSum, 1.0 / words.Count());

            return vecSum;
        }

        public double[] GetWordVector(string word)
        {
            var vecWord = new double[getVectorSize(ModelIndex)];
            getVector(ModelIndex,word, vecWord);
            return vecWord;
        }


        public double[] GetWordDifference(string wordA, string wordB)
        {
            return Add(GetWordVector(wordA), Multiply(GetWordVector(wordB),-1.0));
        }

        public double GetWordSimilarity(string wordA, string wordB)
        {
            return CalculateCosineSimilarity(GetWordVector(wordA), GetWordVector(wordB));
        }

        public double GetWordSimilarity(string wordA, double[] vector)
        {
            return CalculateCosineSimilarity(GetWordVector(wordA), vector);
        }

        public static double CalculateCosineSimilarity(double[] vecA, double[] vecB)
        {
            return DotProduct(vecA, vecB) / (Magnitude(vecA) * Magnitude(vecB));
        }

        public static double DotProduct(double[] vecA, double[] vecB)
        {
            if(vecA.Length != vecB.Length) { throw new Exception("Invalid vector input size"); }

            double dotProduct = 0;

            for (var i = 0; i < vecA.Length; i++)
            {
                dotProduct += (vecA[i] * vecB[i]);
            }

            return dotProduct;
        }

        public static double[] Add(double[] vecA, double[] vecB)
        {
            var vecS = new double[vecA.Length];

            for (var i = 0; i < vecA.Length; i++)
            {
                vecS[i] = vecA[i] + vecB[i];
            }

            return vecS;
        }

        public static double[] Multiply(double[] vecA, double f)
        {
            var vecM = new double[vecA.Length];

            for (var i = 0; i < vecA.Length; i++)
            {
                vecM[i] = vecA[i] * f;
            }

            return vecM;
        }

        public static double Magnitude(double[] vector)
        {
            return Math.Sqrt(DotProduct(vector, vector));
        }
    }

}


