package ru.khtu.service;

import lombok.extern.slf4j.Slf4j;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.stereotype.Service;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.deeplearning4j.nn.weights.WeightInit;

@Service
@Slf4j
public class DeeplearningServiceImpl implements DeeplearningService {

    private final int FEATURES_COUNT = 4, CLASSES_COUNT = 3;
    private DataSet allData;
    private MultiLayerConfiguration multiLayerConfiguration;
    private DataSet trainingData, testData;
    private MultiLayerNetwork multiLayerNetwork;
    private INDArray layerOutput;

    @Override
    public void preparingDataSet() {
        final String FILE_NAME = "iris.csv";
        try (RecordReader recordReader = new CSVRecordReader(0, ',')) {
            recordReader
                    .initialize(
                            new FileSplit(
                                    new ClassPathResource(FILE_NAME)
                                            .getFile() ) );
            DataSetIterator iterator = new RecordReaderDataSetIterator(
                    recordReader, 150, FEATURES_COUNT, CLASSES_COUNT);
            allData = iterator.next();
            log.info("DataSet loaded from : " + FILE_NAME);
        } catch (Exception e) {
            log.warn("Brake with Exception");
        }
    }

    @Override
    public void dataNormalizingAndSplitting() {
        final double QUANTITY_RATIO = 0.65;
        allData.shuffle(/*42*/);
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(allData);
        normalizer.transform(allData);
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(QUANTITY_RATIO);
        log.info("Data shuffled and normalized");
        trainingData = testAndTrain.getTrain();
        testData = testAndTrain.getTest();
        log.info("Prepared training and test data with quantity ratio: " + QUANTITY_RATIO);
    }

    @Override
    public void preparingNetworkMultiLayer() {
        final int l1 = 0, l2 = 1, l3 = 2;
        final Activation TANGENT_HYPERBOLIC = Activation.TANH;
        multiLayerConfiguration = new NeuralNetConfiguration.Builder()
                .iterations(1000)
                .activation(TANGENT_HYPERBOLIC)
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.1)
                .regularization(true).l2(0.0001)
                .list()
                .layer(
                        l1,
                        new DenseLayer
                                .Builder()
                                .nIn(FEATURES_COUNT)
                                .nOut(3)
                                .build() )
                .layer(
                        l2,
                        new DenseLayer
                                .Builder()
                                .nIn(3)
                                .nOut(3)
                                .build())
                .layer(
                        l3,
                        new OutputLayer
                                .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .activation(Activation.SOFTMAX)
                                .nIn(3)
                                .nOut(CLASSES_COUNT)
                                .build())
                .backprop(true)
                .pretrain(false)
                .build();
        log.info("Network MultiLayer configure with activation functions: " + TANGENT_HYPERBOLIC);
    }

    @Override
    public void networkCreatingAndTraining() {
        multiLayerNetwork = new MultiLayerNetwork(multiLayerConfiguration);
        multiLayerNetwork.init();
        multiLayerNetwork.fit(trainingData);
        log.info("Network created and trained");
    }

    @Override
    public void testNetwork() {
        final int CLASS_COUNT = 3;
        log.info("Testing Network for " + CLASS_COUNT + " classes");
        layerOutput = multiLayerNetwork.output(testData.getFeatureMatrix());
        Evaluation evaluation = new Evaluation(CLASS_COUNT);
        evaluation.eval(testData.getLabels(), layerOutput);
        log.info("Testing network completed with result:\n" + evaluation.stats());
    }

}
