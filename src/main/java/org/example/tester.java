package org.example;

import java.util.ArrayList;
import java.util.List;

public class tester {
    static int k=2;
    static int dataLength=171;
    public static void main(String[] args) {
        double[]particle=new double[171];
        for (int i=0;i<171;i++){
            particle[i]=i;
        }
        decode(particle);
    }
    private static List<double[]> decode(double[] particle) {
        List<double[]> centroids = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            double[] centroid = new double[dataLength];
            int offset = i * dataLength;
            for (int j = 0; j < dataLength; j++) {
                System.out.println("offset: "+offset+" j: "+j);
                centroid[j] = particle[offset + j];
            }
            centroids.add(centroid);
        }
        return centroids;
    }
}
