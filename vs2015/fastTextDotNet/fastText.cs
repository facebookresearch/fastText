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
    public class fastText
    {
        [DllImport("fastText.dll", EntryPoint = "initialize", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        private static extern void initialize();

        [DllImport("fastText.dll", EntryPoint = "dispose", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        private static extern void dispose();

        [DllImport("fastText.dll", EntryPoint = "loadModel", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        private static extern bool loadModel([MarshalAs(UnmanagedType.LPStr)] string filename);

        [DllImport("fastText.dll", EntryPoint = "predict", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        private static extern void predict(StringBuilder text, int k);

        [DllImport("fastText.dll", EntryPoint = "getPredictionBufferSize", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I4)]
        private static extern int getPredictionBufferSize();

        [DllImport("fastText.dll", EntryPoint = "getPrediction", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        private static extern void getPrediction(StringBuilder output);

        [DllImport("fastText.dll", EntryPoint = "getVector", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        private static extern void getVector([MarshalAs(UnmanagedType.LPStr)] string word, double[] output);

        [DllImport("fastText.dll", EntryPoint = "getVectorSize", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I4)]
        private static extern int getVectorSize();


        public static bool InitializeWithModel(string filename)
        {
            initialize();
            return loadModel(filename);
        }

        public static void Release()
        {
            dispose();
        }

        public static List<string> Punctuation {get;set;} = new List<string>() { " ", ".", ",", ";", ":", "-", "/", @"\", "!", "?", "'", "\"", "[", "]", "{", "}", "(", ")", "<", ">", "$", "%", "&", "*", "+", "_", "#" };

        public static List<string> StopWords { get; set; } = new List<string>() { "a", "able", "about", "across", "after", "all", "almost", "also", "am", "among", "an", "and", "any", "are", "as", "at", "be", "because", "been", "but", "by", "can", "cannot", "could", "dear", "did", "do", "does", "either", "else", "ever", "every", "for", "from", "get", "got", "had", "has", "have", "he", "her", "hers", "him", "his", "how", "however", "i", "if", "in", "into", "is", "it", "its", "just", "least", "let", "like", "likely", "may", "me", "might", "most", "must", "my", "neither", "no", "nor", "not", "of", "off", "often", "on", "only", "or", "other", "our", "own", "rather", "said", "say", "says", "she", "should", "since", "so", "some", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "tis", "to", "too", "twas", "us", "wants", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would", "yet", "you", "your" };

        public static string GetPrediction(string text, int numberOfTopLabelsToReturn)
        {
            var buffer = new StringBuilder(text + "\n");
            predict(buffer, numberOfTopLabelsToReturn);
            var out_buffer = new StringBuilder(getPredictionBufferSize());
            getPrediction(out_buffer);
            return out_buffer.ToString().TrimEnd(new char[] { '\n', '\r' });
        }

        public static double[] GetParagraphVector(string text)
        {
            var words = text.Split(Punctuation.ToArray(), StringSplitOptions.RemoveEmptyEntries).ToList();
            var vecSum = new double[fastText.getVectorSize()];

            words = words.Except(StopWords).ToList();

            foreach (var word in words)
            {
                var vecWord = GetWordVector(word);
                vecSum = Add(vecSum, vecWord);
            }

            vecSum = Multiply(vecSum, 1.0 / words.Count());

            return vecSum;
        }

        public static double[] GetWordVector(string word)
        {
            var vecWord = new double[fastText.getVectorSize()];
            getVector(word, vecWord);
            return vecWord;
        }



        public static double GetWordSimilarity(string wordA, string wordB)
        {
            return CalculateCosineSimilarity(GetWordVector(wordA), GetWordVector(wordB));
        }

        public static double CalculateCosineSimilarity(double[] vecA, double[] vecB)
        {
            return DotProduct(vecA, vecB) / (Magnitude(vecA) * Magnitude(vecB));
        }

        private static double DotProduct(double[] vecA, double[] vecB)
        {
            if(vecA.Length != vecB.Length) { throw new Exception("Invalid vector input size"); }

            double dotProduct = 0;

            for (var i = 0; i < vecA.Length; i++)
            {
                dotProduct += (vecA[i] * vecB[i]);
            }

            return dotProduct;
        }

        private static double[] Add(double[] vecA, double[] vecB)
        {
            var vecS = new double[vecA.Length];

            for (var i = 0; i < vecA.Length; i++)
            {
                vecS[i] = vecA[i] + vecB[i];
            }

            return vecS;
        }

        private static double[] Multiply(double[] vecA, double f)
        {
            var vecM = new double[vecA.Length];

            for (var i = 0; i < vecA.Length; i++)
            {
                vecM[i] = vecA[i] * f;
            }

            return vecM;
        }

        private static double Magnitude(double[] vector)
        {
            return Math.Sqrt(DotProduct(vector, vector));
        }
    }

}


