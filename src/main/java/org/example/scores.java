package org.example;
import java.util.Arrays;
import java.util.List;

public class scores {
    //越大越好 SC,CH,DUNN
    //越小越好 VE,DBI,
    private static double distance(double[] a, double[] b) {
        double distance = 0;
        for (int i = 0; i < a.length; i++) {
            distance += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(distance);
    }
    public static double daviesBouldinIndex(List<List<double[]>> result) {
        if (result == null)
            return -1;

        int totalClusters = result.size();
        double[] clusterMeanDistances = new double[totalClusters];

        for (int i = 0; i < totalClusters; i++) {
            List<double[]> cluster = result.get(i);
            double[] clusterMean = new double[cluster.get(0).length];
            for (double[] point : cluster) {
                for (int j = 0; j < point.length; j++) {
                    clusterMean[j] += point[j];
                }
            }
            for (int j = 0; j < clusterMean.length; j++) {
                clusterMean[j] /= cluster.size();
            }

            double sum = 0;
            for (double[] point : cluster) {
                sum += distance(point, clusterMean);
            }
            clusterMeanDistances[i] = sum / cluster.size();
        }

        double daviesBouldin = 0;
        for (int i = 0; i < totalClusters; i++) {
            double maxRatios = 0;
            for (int j = 0; j < totalClusters; j++) {
                if (i != j) {
                    double[] mean1 = centroid(result.get(i));
                    double[] mean2 = centroid(result.get(j));
                    double distanceBetweenMeans = distance(mean1, mean2);
                    double ratio = (clusterMeanDistances[i] + clusterMeanDistances[j]) / distanceBetweenMeans;
                    maxRatios = Math.max(maxRatios, ratio);
                }
            }
            daviesBouldin += maxRatios;
        }

        return daviesBouldin / totalClusters;
    }
    private static double[] centroid(List<double[]> cluster) {
        double[] centroid = new double[cluster.get(0).length];
        for (double[] point : cluster) {
            for (int i = 0; i < point.length; i++) {
                centroid[i] += point[i];
            }
        }
        for (int i = 0; i < centroid.length; i++) {
            centroid[i] /= cluster.size();
        }
        return centroid;
    }
    public static double silhouetteCoefficient(List<List<double[]>> result) {
        double sc = 0.0;
        if (result==null)
            return -1;
        for (int i = 0; i < result.size(); i++) {
            List<double[]> resultSubList = result.get(i);
            if (resultSubList==null)
                return -1;
            for (int j = 0; j < resultSubList.size(); j++) {
                double[] e = resultSubList.get(j);
                double a = 0.0;
                double aCountTimes = 0.0;
                double b = 0.0;
                double bCountTimes = 0.0;
                for (double[] doubles : resultSubList) {
                    a += distance(e, doubles);
                    aCountTimes++;
                }
                if (aCountTimes != 0)
                    a /= aCountTimes;
                for (int k = 0; k < result.size(); k++) {
                    if (i == k)
                        continue;
                    List<double[]> differentCluster = result.get(k);
                    for (double[] doubles : differentCluster) {
                        b += distance(doubles, e);
                        bCountTimes++;
                    }
                }
                b /= bCountTimes;
                double s = (b - a) / Math.max(a, b);
                sc += s;
            }
        }
        double n = 0.0;
        for (List<double[]> doubles : result) {
            for (int j = 0; j < doubles.size(); j++) {
                n++;
            }
        }
        sc /= n;
        return sc;
    }
    public static double varianceEvaluation(List<List<double[]>> result) {
        double var = 0.0;
        if (result == null) {
            return -1;
        }
        for (int i = 0; i < result.size(); i++) {
            List<double[]> resultSubList = result.get(i);
            if (resultSubList == null) {
                return -1;
            }
            double[] clusterCenter = new double[resultSubList.get(0).length];
            for (int j = 0; j < clusterCenter.length; j++) {
                double sum = 0.0;
                for (double[] dataPoint : resultSubList) {
                    sum += dataPoint[j];
                }
                clusterCenter[j] = sum / resultSubList.size();
            }
            double s = 0.0;
            for (double[] dataPoint : resultSubList) {
                s += distance(dataPoint, clusterCenter);
            }
            s = Math.pow(s, 2) / resultSubList.size();
            var += s;
        }
        return var;
    }
    public static double dunnIndex(List<List<double[]>> result) {
        if (result == null) {
            return -1;
        }

        double maxDiameter = Double.MIN_VALUE;
        double minDistance = Double.MAX_VALUE;

        // Calculate maximum diameter of clusters
        for (List<double[]> cluster : result) {
            double diameter = 0.0;
            for (int i = 0; i < cluster.size(); i++) {
                for (int j = i + 1; j < cluster.size(); j++) {
                    double distance = distance(cluster.get(i), cluster.get(j));
                    if (distance > diameter) {
                        diameter = distance;
                    }
                }
            }
            if (diameter > maxDiameter) {
                maxDiameter = diameter;
            }
        }

        // Calculate minimum distance between clusters
        for (int i = 0; i < result.size(); i++) {
            List<double[]> cluster1 = result.get(i);
            for (int j = i + 1; j < result.size(); j++) {
                List<double[]> cluster2 = result.get(j);
                for (double[] point1 : cluster1) {
                    for (double[] point2 : cluster2) {
                        double distance = distance(point1, point2);
                        if (distance < minDistance) {
                            minDistance = distance;
                        }
                    }
                }
            }
        }

        return minDistance / maxDiameter;
    }
    public static double calinskiHarabaszIndex(List<List<double[]>> result) {
        if (result==null)
            return -1;
        // 计算聚类结果中所有点的总数
        int totalPoints = result.stream().mapToInt(List::size).sum();
        // 如果总点数小于等于聚类数，Calinski-Harabasz指数无法计算
        if (totalPoints <= result.size()) {
            throw new IllegalArgumentException("Cannot compute Calinski-Harabasz index when total number of points is less than or equal to the number of clusters.");
        }
        // 计算所有点的平均值
        if (result.get(0).isEmpty())
            return -1;
        double[] totalMean = new double[result.get(0).get(0).length];
        for (List<double[]> cluster : result) {
            if (cluster.isEmpty())
                return -1;
            for (double[] point : cluster) {
                for (int i = 0; i < point.length; i++) {
                    totalMean[i] += point[i];
                }
            }
        }
        for (int i = 0; i < totalMean.length; i++) {
            totalMean[i] /= totalPoints;
        }
        // 计算聚类内部的平均距离平方和
        double[] intraClusterVariances = new double[result.size()];
        for (int i = 0; i < result.size(); i++) {
            List<double[]> cluster = result.get(i);
            double[] clusterMean = new double[cluster.get(0).length];
            for (double[] point : cluster) {
                for (int j = 0; j < point.length; j++) {
                    clusterMean[j] += point[j];
                }
            }
            for (int j = 0; j < clusterMean.length; j++) {
                clusterMean[j] /= cluster.size();
            }
            double sum = 0;
            for (double[] point : cluster) {
                sum += Math.pow(distance(point, clusterMean), 2);
            }
            intraClusterVariances[i] = sum / cluster.size();
        }
        // 计算聚类间的平均距离平方和
        double interClusterVariance = 0;
        for (List<double[]> cluster : result) {
            double[] clusterMean = new double[cluster.get(0).length];
            for (double[] point : cluster) {
                for (int i = 0; i < point.length; i++) {
                    clusterMean[i] += point[i];
                }
            }
            for (int i = 0; i < clusterMean.length; i++) {
                clusterMean[i] /= cluster.size();
            }
            interClusterVariance += cluster.size() * Math.pow(distance(clusterMean, totalMean), 2);
        }
        interClusterVariance /= result.size() - 1;
        return interClusterVariance / Arrays.stream(intraClusterVariances).sum() * (totalPoints - result.size()) / (result.size() - 1);
    }
}
