# src/main.py

from src.config            import RANDOM_STATE
from src.utils             import add_age_quartile, add_log_cpk
from src.data_loader       import load_data, split_data
from src.preprocessing     import build_preprocessor
from src.modeling          import tune_and_train
from src.evaluation        import compute_metrics, plot_roc, plot_calibration, plot_confusion
from src.survival_analysis import km_plot, cox_model
from src.explainability    import shap_summary, lime_explain
from src.dashboard         import run_dashboard

def main():
    # 1) Load & engineer
    df = load_data()
    df = add_age_quartile(df, n_quartiles=4)
    df = add_log_cpk(df)

    # 2) Split
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)

    # 3) Preprocessor
    preprocessor = build_preprocessor()

    # 4) Tune & train
    uncal_pipe, cal_pipe, study = tune_and_train(X_train, y_train, preprocessor)

    # 5) Evaluate
    metrics = compute_metrics(cal_pipe, X_test, y_test)
    fig_roc       = plot_roc(cal_pipe, X_test, y_test)
    fig_cal       = plot_calibration(cal_pipe, X_test, y_test)
    fig_confusion = plot_confusion(cal_pipe, X_test, y_test)

    # 6) Survival analysis
    fig_km = km_plot(df, time_col="time", event_col="death_event", group_col="ejection_fraction")
    cph    = cox_model(df)

    # 7) Explainability
    fig_shap = shap_summary(uncal_pipe, X_train, X_test, top_n=10)
    fig_lime = lime_explain(uncal_pipe, X_train, X_test, idx=0)

    # 8) Launch dashboard
    run_dashboard(
        metrics,
        {
          'roc': fig_roc,
          'calibration': fig_cal,
          'confusion': fig_confusion,
          'km': fig_km,
          'shap': fig_shap,
          'lime': fig_lime,
          'styles': ['https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css']
        }
    )

if __name__ == "__main__":
    main()
