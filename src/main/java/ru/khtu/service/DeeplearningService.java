package ru.khtu.service;

public interface DeeplearningService {

    void preparingDataSet();

    void dataNormalizingAndSplitting();

    void preparingNetworkMultiLayer();

    void networkCreatingAndTraining();

    void testNetwork();

}
