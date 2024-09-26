package ru.khtu.component;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;
import ru.khtu.service.DeeplearningService;

@Slf4j
@Component
@RequiredArgsConstructor
public class Runner implements CommandLineRunner {

    private final DeeplearningService deeplearningService;

    @Override
    public void run(String... args) throws Exception {
        log.info("Run CommandLine Runner Component ...");
        try {
            deeplearningService.preparingDataSet();
            deeplearningService.dataNormalizingAndSplitting();
            deeplearningService.preparingNetworkMultiLayer();
            deeplearningService.networkCreatingAndTraining();
            deeplearningService.testNetwork();
            log.info("CommandLine Runner running complete");
        } catch (Exception e) {
            log.warn("Brake with Exception");
        }
    }

}
