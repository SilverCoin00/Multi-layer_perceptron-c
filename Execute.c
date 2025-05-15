#include "Core.h"

int main() {
    char file[] = "flag.csv";
    Data_Frame* df = read_csv(file, 1000, ",");
    Dataset_2* full_ds = trans_dframe_to_dset2(df, "t");
    Dataset_2* ds = (Dataset_2*)malloc(sizeof(Dataset_2));
    Dataset_2* test_ds = (Dataset_2*)malloc(sizeof(Dataset_2));
    free_data_frame(df);
    train_test_split_ds2(full_ds, ds, test_ds, 0.05, 0);
    free_dataset2(full_ds);
    Standard_scaler* scaler = (Standard_scaler*)malloc(sizeof(Standard_scaler));
    scaler_fit(ds->x, NULL, ds->samples, ds->features, scaler, "Standard_scaler");
    scaler_transform(ds->x, NULL, ds->samples, ds->features, scaler, "Standard_scaler");
    scaler_transform(test_ds->x, NULL, test_ds->samples, test_ds->features, scaler, "Standard_scaler");
    free_scaler(scaler, "Standard_scaler");

    MLP* model = new_sequential_model((int[]) {ds->samples, ds->features}, 
                            (Keras_layer*[]){&(Keras_layer){&(Dense){16, "relu"}, NULL}, 
                                            &(Keras_layer){NULL, &(Dropout){0.2}},
                                            &(Keras_layer){&(Dense){8, "relu"}, NULL}, 
                                            &(Keras_layer){NULL, &(Dropout){0.2}}, 
                                            &(Keras_layer){&(Dense){3, "softmax"}, NULL}, NULL});
    //Optimizer* opt = new_optimizer("SGD", 0.008, 0.9, 1, Nan, Nan, Nan, Nan, Nan);
    //Optimizer* opt = new_optimizer("Adagrad", 0.06, Nan, Nan, Nan, Nan, Nan, 1e-8, 1e-2);
    //Optimizer* opt = new_optimizer("RMSProp", 0.002, Nan, Nan, Nan, Nan, 0.9, 1e-8, Nan);
    //Optimizer* opt = new_optimizer("Adadelta", 0.0001, Nan, Nan, Nan, Nan, 0.9, 1e-8, Nan);
    Optimizer* opt = new_optimizer("Adam", 0.003, Nan, Nan, 0.9, 0.999, Nan, 1e-8, Nan);
    Early_Stopping* estop = new_earlystopping("val_accuracy", 1e-8, 7, Nan, 1);
    model_compile(model, opt, "categorical_crossentropy", "categorical_accuracy");
    model_fit(model, ds, 350, 32, 0.2, estop);

    model_evaluate(model, test_ds);
    free_mlp_model(model);
    free(estop);
    free_dataset2(ds);
    free_dataset2(test_ds);
    return 0;
}
