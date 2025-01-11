#pragma once
#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <chrono>
#include <string>
#include <cstring>
#include <stdlib.h>
#include <omp.h>
#include <climits>
#include <cmath>
using namespace std;

#define RUN_MNIST 0
#define RUN_CIFAR10 1

string CIFAR10_file[5] = {"./cifar-10/data_batch_1.bin","./cifar-10/data_batch_2.bin","./cifar-10/data_batch_3.bin",
                          "./cifar-10/data_batch_4.bin","./cifar-10/data_batch_5.bin"};

#define THRESHOLD 0.001

typedef struct point {
    int clusterId = 0;
    int label;
    vector <float> data;

    vector <float> getData() { return data; }
} Point;

typedef struct cluster {
    int numPoint = 0;
    vector <Point> dataPoint;
    int label;
} Cluster;

void printPoint(Point point)
{
    int size = 32;
    cout << "Label: " << point.label << endl;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
            cout << point.data[i * size + j] << " ";
        cout << endl;
    }
}

float distance(vector<float> point1, vector<float> point2, int dim)
{
    float totalDistance = 0;
    for (int i = 0; i < dim; i++)
    {
        totalDistance += (point1[i] - point2[i]) * (point1[i] - point2[i]);
    }

    return totalDistance;
}

class KMeans
{
public:
    int numClusters;            //nombre de centroïds
    int dim;                    //dim d'un point du dataset
    int numPoints;

    ofstream fileO;

    vector <Point> dataPoints;

    vector <vector<float>> centrePoints;
    vector <int> labels;
    vector <Cluster> cluster;

    // _k as kmeans, _dim in dimension
    KMeans(int _k, int _dim);
    void readDataFromFile(string dir="");
    void initCentroid();
    void process();
    void printData();
    void printCentre();
    void printResult();
    void checkAccuracy();
    void DaviesBouldinIndex();

private:
    float threshold = 0.01;
    void reset();
};

KMeans::KMeans(int _k, int _dim)
{
    numClusters = _k;
    dim = _dim;
    numPoints = 0;
    cluster.resize(numClusters + 1);
    cout << "Nombre de clusters: " << numClusters << endl;
    cout << "Data point dimension: " << dim << endl;

    fileO.open("output.txt");
    fileO << numClusters << endl;
    fileO << dim << endl;
}

void KMeans::printData()
{
    int idx = 0;
    cout << "----Current number of data points:" << dataPoints.size() << "----\n";
    cout << "----List of Data Points: ----" << endl;
    for (auto point : dataPoints)
    {
        for (int j = 0; j < dim; j++)
            cout << point.data[j] << " ";
        cout << endl;
    }
}

void KMeans::printCentre()
{
    cout << "Centre points:\n";
    for (auto point : centrePoints)
    {
        for (int i = 0; i < dim; i++)
        {
            fileO << point[i] << " ";
            cout << point[i] << " ";
        }
        cout << endl; fileO << endl;
    }
}

void KMeans::reset()
{
    for (int i = 0; i < numClusters; i++)
    {
        cluster[i].numPoint = 0;
        cluster[i].dataPoint.clear();
    }
}

void KMeans::readDataFromFile(string dir)
{
    ifstream file3;
    int numFiles = 5;
    int numDataPerFiles = 10000;
 
    for (int file_idx = 0; file_idx < numFiles; file_idx++)
    {
        file3.open(CIFAR10_file[file_idx], ios::binary);
        if (file3.is_open())
        {
            cout << "fichier ouvert! \n";
            unsigned char label = 0;
            for (int cnt = 0; cnt < numDataPerFiles; cnt++)
            {
                Point point;
                file3.read((char*)&label, sizeof(label));
                // cout << "Label: " << (int)label << endl;
                point.label = (int)label;
                for (int i = 0; i < dim; ++i)
                {
                    // if (i % 32 == 0) cout << endl;
                    unsigned char pixel = 0;
                    file3.read((char*)&pixel, sizeof(pixel));
                    // cout << (int)pixel << " ";
                    point.data.push_back((int)pixel);
                }
                dataPoints.push_back(point);
            }
        }
        file3.close();
    }

    cout << "Lecture des fichiers terminée!\n";
    numPoints = dataPoints.size();
}

void KMeans::initCentroid()
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    static std::default_random_engine generator(123);
    int end = numPoints - 1;
    std::uniform_int_distribution<int> distribution(0, end);

    vector<int> p(numPoints);
    fill(p.begin(), p.end(), 0);
    int cnt = 0;
    cout << "générer des centroïds de 0 à " << end << endl;
    cout << "Liste des centroids:\n";
    while (cnt < numClusters)
    {
        int number = distribution(generator);
        if (!p[number])
        {
            cout << number << " ";
            p[number] = 1;
            cnt++;
            centrePoints.push_back(dataPoints[number].data);
        }

    }
    cout << endl;
}

void div(vector<float> &p, float x, int dim)
{
    for (int i = 0; i < dim; i++)
    {
        p[i] = p[i] / x;
    }
}

void printArr(float* p, int dim)
{
    cout << "array: ";
    for (int i = 0; i < dim; i++)
        cout << p[i] << " ";
    cout << endl;
}

void KMeans::process()
{
    cout << "Nombre total des points: " << numPoints << endl;
    while (true)
    {
        float th = 0;
        reset();
        // Calculer les distances
        for (int i = 0; i < numPoints; i++)
        {
            float dmin = INT_MAX;
            int cen = 0;
            for (int j = 0; j < numClusters; j++)
            {
                float d = distance(dataPoints[i].data, centrePoints[j], dim);
                if (d < dmin)
                {
                    dmin = d;
                    cen = j;
                }
            }
            dataPoints[i].clusterId = cen;
        }
        
        // Affecter des points aux clusters
        for (int i = 0; i < numPoints; i++)
        {
            cluster[dataPoints[i].clusterId].dataPoint.push_back(dataPoints[i]);
            cluster[dataPoints[i].clusterId].numPoint++;
        }

        //  Recalculer les centres des clusters
        for (int i = 0; i < numClusters; i++)
        {
            int clusterSize = cluster[i].numPoint;
            if (clusterSize == 0) continue;
            for (int idx = 0; idx < dim; idx++)
            {
                double sum = 0.0;
                double oldValue = centrePoints[i][idx];

                for (int j = 0; j < clusterSize; j++)
                    sum += cluster[i].dataPoint[j].data[idx];
                centrePoints[i][idx] = sum / clusterSize;
                th += abs(centrePoints[i][idx] - oldValue);
            }
        }
        if (th < THRESHOLD) break;
        
    }
}

void KMeans::printResult()
{
    for (auto point : centrePoints)
    {
        for (int i = 0; i < dim; ++i)
            fileO << point[i] << " ";
        fileO << endl;
    }

    for (int i = 0; i < numClusters; i++)
    {
        fileO << cluster[i].numPoint << endl;
        if (cluster[i].numPoint == 0) { cout << "(pas de données dans ce cluster !)\n"; continue; }
        for (auto point : cluster[i].dataPoint)
        {
            //cout << point << " ";
            for (int j = 0; j < dim; j++)
            {
                fileO << point.data[j] << " ";
            }
            fileO << endl;
        }
    }
    fileO.close();
}

void KMeans::checkAccuracy()
{
    for (int i=0; i< numClusters ; i++)
    {
        vector <int> cnt(10);
        int label = 0 , max = INT_MIN;
        for (auto point : cluster[i].dataPoint)
        {
            cnt[point.label]++;
            if (cnt[point.label] > max)
            {
                max = cnt[point.label];
                label = point.label;
            }
        }
        cluster[i].label = label;
    }

    ofstream checkFile;
    checkFile.open("check_CIFAR10.txt");
    for (auto point : dataPoints)
    {
        checkFile << "point label is: " << point.label << " - point cluster is: " << point.clusterId << " - this cluster label is: " << cluster[point.clusterId].label << endl; //fichier des prédictions
    }
    checkFile.close();

    int i = 0;
    for (auto _cluster : cluster)
    {
        cout << "Cluster " << i << " true label : " << _cluster.label << endl;
        i++;
    }
    int cnt = 0;

    for (auto point : dataPoints)
    {
        if (point.label == cluster[point.clusterId].label)
            cnt++;
    }

    cout << "Accuracy: " << (float) cnt / numPoints * 100.0 << endl;
    cout << cnt << " sur " << numPoints << endl;

    //printPoint(dataPoints[1]);
}

void KMeans::DaviesBouldinIndex()
{
    vector <float> IntraClusterDispersion(10);
    for (int i = 0; i < numClusters; i++)
    {
        int clusterIdx = i;
        for (int j = 0; j < numPoints; j++)
            if (dataPoints[j].clusterId == clusterIdx)
            {
                IntraClusterDispersion[clusterIdx] += distance(dataPoints[j].data, centrePoints[i], dim);
            }
    }

    float ClusterSeperationMeasure[10][10];
    memset(ClusterSeperationMeasure, 0, sizeof(ClusterSeperationMeasure));

    for (int i = 0; i < numClusters; i++)
        for (int j = i; j < numClusters; j++)
            ClusterSeperationMeasure[i][j] = distance(centrePoints[i], centrePoints[j], dim);

    float SimilarityBetweenClusters[10][10];
    for (int i = 0; i < numClusters; i++)
        for (int j = i; j < numClusters; j++)
        {
            SimilarityBetweenClusters[i][j] = (IntraClusterDispersion[i] + IntraClusterDispersion[j]) / ClusterSeperationMeasure[i][j];
        }

    vector <float> MostSimilarCluster(10);
    for (int i = 0; i < numClusters - 1; i++)
    {
        float max = INT_MIN;
        for (int j = i + 1; j < numClusters; j++)
        {
            if (max < SimilarityBetweenClusters[i][j])
                max = SimilarityBetweenClusters[i][j];
        }

        MostSimilarCluster[i] = max;
    }

    float result = 0;
    for (int i = 0; i < numClusters; i++)
    {
        result += MostSimilarCluster[i];
    }

    result = result / numClusters;
    cout << "Davies - Bouldin Index: " << result << endl;

}