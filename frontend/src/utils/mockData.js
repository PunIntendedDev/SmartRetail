// Mock data for SmartRetail AI Customer Analytics Dashboard

export const quickStats = {
  totalCustomers: "3,312",
  totalFeatures: "13",
  bestClassifier: "MLP Classifier (LDA)",
  bestRegressor: "MLP Regressor",
  classificationAccuracy: "100.0%",
  regressionR2: "0.6593",
};

export const pipelinePhases = [
  { id: 1, name: "Dataset Ingestion", desc: "UCI Online Retail Ledger (541k records)" },
  { id: 2, name: "Data Cleaning", desc: "Dropped duplicates, cancelled orders, and null CustomerIDs" },
  { id: 3, name: "Feature Engineering", desc: "R, F, M, average spend, and basket sizes strictly on Months 1-9" },
  { id: 4, name: "Standard Scaling", desc: "Fitted StandardScaler strictly on training set customer columns" },
  { id: 5, name: "PCA & LDA Projections", desc: "Drop multicollinear spent; project PCA (93% var) and separation LDA" },
  { id: 6, name: "Machine Learning", desc: "GridSearchCV Tuning, MLPClassifier & Regressor training" },
  { id: 7, name: "Inference & Serving", desc: "Load joblib binaries offline for dynamic forecasting" },
];

export const recentActivity = [
  { time: "Just now", event: "API pipeline listener instantiated in routes/api" },
  { time: "1 hour ago", event: "Calculated validation F1-score comparisons for MLP vs Logistic Regression" },
  { time: "Yesterday", event: "Saved pre-fit PCA component mapping joblib binaries" },
  { time: "2 days ago", event: "Resolved Category composition perfect multicollinearity by dropping Other_Spend_Pct" },
  { time: "4 days ago", event: "Ingested and structured splits at customer level using random seed 42" },
];

export const classificationComparisons = [
  { space: "Raw Scaled", model: "Logistic Regression", accuracy: "94.56%", precision: "89.23%", recall: "84.34%", f1: "0.8672", auc: "0.9654" },
  { space: "Raw Scaled", model: "MLP Classifier", accuracy: "97.12%", precision: "94.10%", recall: "92.35%", f1: "0.9321", auc: "0.9876" },
  { space: "PCA Components", model: "Logistic Regression", accuracy: "93.18%", precision: "87.05%", recall: "80.45%", f1: "0.8362", auc: "0.9412" },
  { space: "PCA Components", model: "MLP Classifier", accuracy: "95.62%", precision: "91.80%", recall: "88.10%", f1: "0.8991", auc: "0.9711" },
  { space: "LDA Projections", model: "Logistic Regression", accuracy: "98.85%", precision: "97.20%", recall: "97.20%", f1: "0.9720", auc: "0.9991" },
  { space: "LDA Projections", model: "MLP Classifier", accuracy: "100.0%", precision: "100.0%", recall: "100.0%", f1: "1.0000", auc: "1.0000", isBest: true },
];

export const regressionComparisons = [
  { model: "Linear Regression", mse: "86,432,907.31", rmse: "$9,296.93", r2: "-0.0022" },
  { model: "MLP Regressor", mse: "86,051,988.12", rmse: "$9,276.42", r2: "0.0022", isBest: true },
];

export const featureImportance = [
  { name: "Product Diversity", importance: 1.1471 },
  { name: "Average Spend", importance: 0.8780 },
  { name: "Recency", importance: -0.7460 },
  { name: "Average Qty / Order", importance: 0.5863 },
  { name: "Frequency", importance: 0.5589 },
  { name: "Total Orders", importance: 0.5589 },
  { name: "Monetary", importance: -0.4183 },
  { name: "Avg Basket Size", importance: -0.2812 },
  { name: "Stationery Spend %", importance: -0.2071 },
  { name: "Decorations Spend %", importance: -0.1600 },
  { name: "Gadgets Spend %", importance: -0.1199 },
  { name: "Homeware Spend %", importance: -0.0142 },
  { name: "Kitchenware Spend %", importance: -0.0048 },
];

export const pcaScreeData = [
  { name: "PC1", ratio: 0.2586, cumulative: 0.2586 },
  { name: "PC2", ratio: 0.2315, cumulative: 0.4901 },
  { name: "PC3", ratio: 0.2306, cumulative: 0.7207 },
  { name: "PC4", ratio: 0.2119, cumulative: 0.9326 },
  { name: "PC5", ratio: 0.0674, cumulative: 1.0000 },
];

export const sampleDataset = [
  { customerId: 17850, recency: 372, frequency: 34, monetary: 5391.21, averageSpend: 158.57, diversity: 21, totalOrders: 34, basketSize: 8.5, qtyPerOrder: 52.4, isHighValue: 1, futureSpend: 0.0 },
  { customerId: 13047, recency: 31, frequency: 10, monetary: 3232.59, averageSpend: 323.26, diversity: 18, totalOrders: 10, basketSize: 4.2, qtyPerOrder: 30.1, isHighValue: 1, futureSpend: 1425.33 },
  { customerId: 12583, recency: 2, frequency: 15, monetary: 6205.42, averageSpend: 413.70, diversity: 35, totalOrders: 15, basketSize: 12.0, qtyPerOrder: 98.4, isHighValue: 1, futureSpend: 2315.55 },
  { customerId: 13748, recency: 95, frequency: 5, monetary: 948.20, averageSpend: 189.64, diversity: 12, totalOrders: 5, basketSize: 3.1, qtyPerOrder: 22.0, isHighValue: 0, futureSpend: 0.0 },
  { customerId: 15100, recency: 330, frequency: 3, monetary: 876.00, averageSpend: 292.00, diversity: 2, totalOrders: 3, basketSize: 1.0, qtyPerOrder: 15.0, isHighValue: 0, futureSpend: 0.0 },
  { customerId: 15291, recency: 25, frequency: 15, monetary: 4596.51, averageSpend: 306.43, diversity: 28, totalOrders: 15, basketSize: 6.8, qtyPerOrder: 45.2, isHighValue: 1, futureSpend: 1250.00 },
  { customerId: 14688, recency: 7, frequency: 21, monetary: 5122.95, averageSpend: 243.95, diversity: 42, totalOrders: 21, basketSize: 8.2, qtyPerOrder: 64.0, isHighValue: 1, futureSpend: 1980.42 },
  { customerId: 17809, recency: 16, frequency: 12, monetary: 4627.62, averageSpend: 385.64, diversity: 15, totalOrders: 12, basketSize: 5.4, qtyPerOrder: 32.5, isHighValue: 1, futureSpend: 825.00 },
  { customerId: 15311, recency: 0, frequency: 91, monetary: 58352.77, averageSpend: 641.24, diversity: 92, totalOrders: 91, basketSize: 22.5, qtyPerOrder: 195.4, isHighValue: 1, futureSpend: 15234.80 },
  { customerId: 14527, recency: 2, frequency: 55, monetary: 7824.30, averageSpend: 142.26, diversity: 48, totalOrders: 55, basketSize: 9.8, qtyPerOrder: 58.1, isHighValue: 1, futureSpend: 2430.22 },
  { customerId: 16029, recency: 38, frequency: 63, monetary: 81024.84, averageSpend: 1286.11, diversity: 55, totalOrders: 63, basketSize: 18.2, qtyPerOrder: 250.0, isHighValue: 1, futureSpend: 18230.15 },
  { customerId: 13408, recency: 1, frequency: 62, monetary: 28117.04, averageSpend: 453.50, diversity: 32, totalOrders: 62, basketSize: 11.2, qtyPerOrder: 112.5, isHighValue: 1, futureSpend: 6250.40 },
  { customerId: 13767, recency: 2, frequency: 24, monetary: 17228.50, averageSpend: 717.85, diversity: 18, totalOrders: 24, basketSize: 14.5, qtyPerOrder: 158.0, isHighValue: 1, futureSpend: 4210.50 },
  { customerId: 12748, recency: 0, frequency: 210, monetary: 33719.73, averageSpend: 160.57, diversity: 125, totalOrders: 210, basketSize: 32.4, qtyPerOrder: 142.1, isHighValue: 1, futureSpend: 9235.10 },
  { customerId: 15502, recency: 15, frequency: 4, monetary: 1530.12, averageSpend: 382.53, diversity: 10, totalOrders: 4, basketSize: 3.0, qtyPerOrder: 20.0, isHighValue: 0, futureSpend: 250.00 },
  { customerId: 16938, recency: 250, frequency: 1, monetary: 240.50, averageSpend: 240.50, diversity: 5, totalOrders: 1, basketSize: 5.0, qtyPerOrder: 12.0, isHighValue: 0, futureSpend: 0.0 },
  { customerId: 17001, recency: 80, frequency: 2, monetary: 480.00, averageSpend: 240.00, diversity: 8, totalOrders: 2, basketSize: 4.0, qtyPerOrder: 18.0, isHighValue: 0, futureSpend: 0.0 },
  { customerId: 12346, recency: 325, frequency: 1, monetary: 77183.60, averageSpend: 77183.60, diversity: 1, totalOrders: 1, basketSize: 1.0, qtyPerOrder: 74215.0, isHighValue: 1, futureSpend: 0.0 },
  { customerId: 12347, recency: 2, frequency: 7, monetary: 4310.00, averageSpend: 615.71, diversity: 24, totalOrders: 7, basketSize: 15.2, qtyPerOrder: 85.3, isHighValue: 1, futureSpend: 1230.00 },
  { customerId: 12348, recency: 75, frequency: 4, monetary: 1797.24, averageSpend: 449.31, diversity: 8, totalOrders: 4, basketSize: 5.5, qtyPerOrder: 38.0, isHighValue: 1, futureSpend: 0.0 },
];

export const pcaScatterData = [
  // High value customers (class 1)
  { pc1: 1.8, pc2: 2.1, isHighValue: 1 },
  { pc1: 2.2, pc2: 0.8, isHighValue: 1 },
  { pc1: 0.9, pc2: 1.7, isHighValue: 1 },
  { pc1: 1.5, pc2: 1.2, isHighValue: 1 },
  { pc1: 2.5, pc2: 2.4, isHighValue: 1 },
  { pc1: 3.0, pc2: -0.5, isHighValue: 1 },
  { pc1: 0.5, pc2: 2.9, isHighValue: 1 },
  { pc1: 1.2, pc2: -1.1, isHighValue: 1 },
  { pc1: -0.2, pc2: 1.8, isHighValue: 1 },
  // Low value customers (class 0)
  { pc1: -1.5, pc2: -0.8, isHighValue: 0 },
  { pc1: -2.1, pc2: 0.4, isHighValue: 0 },
  { pc1: -0.8, pc2: -1.2, isHighValue: 0 },
  { pc1: -1.2, pc2: 0.9, isHighValue: 0 },
  { pc1: -0.5, pc2: -0.2, isHighValue: 0 },
  { pc1: 0.2, pc2: -1.5, isHighValue: 0 },
  { pc1: -1.8, pc2: -2.0, isHighValue: 0 },
  { pc1: -2.4, pc2: -0.5, isHighValue: 0 },
  { pc1: 0.8, pc2: -0.9, isHighValue: 0 },
];

export const ldaScatterData = [
  // Class 1 (High Value)
  { id: 1, ld1: 3.5, y: 0.2, isHighValue: 1 },
  { id: 2, ld1: 2.8, y: 0.1, isHighValue: 1 },
  { id: 3, ld1: 4.1, y: -0.1, isHighValue: 1 },
  { id: 4, ld1: 3.2, y: 0.3, isHighValue: 1 },
  { id: 5, ld1: 2.5, y: -0.2, isHighValue: 1 },
  { id: 6, ld1: 4.8, y: 0.0, isHighValue: 1 },
  { id: 7, ld1: 3.0, y: -0.3, isHighValue: 1 },
  // Class 0 (Low Value)
  { id: 8, ld1: -2.1, y: 0.1, isHighValue: 0 },
  { id: 9, ld1: -1.8, y: -0.2, isHighValue: 0 },
  { id: 10, ld1: -3.5, y: 0.3, isHighValue: 0 },
  { id: 11, ld1: -2.7, y: -0.1, isHighValue: 0 },
  { id: 12, ld1: -1.2, y: 0.2, isHighValue: 0 },
  { id: 13, ld1: -4.0, y: 0.0, isHighValue: 0 },
  { id: 14, ld1: -2.4, y: -0.3, isHighValue: 0 },
];

export const predictedVsActualData = [
  { actual: 1200, predicted: 1150 },
  { actual: 2315, predicted: 2420 },
  { actual: 950, predicted: 910 },
  { actual: 0, predicted: 45 },
  { actual: 4596, predicted: 4320 },
  { actual: 1980, predicted: 2150 },
  { actual: 825, predicted: 760 },
  { actual: 15234, predicted: 14890 },
  { actual: 2430, predicted: 2310 },
  { actual: 18230, predicted: 19100 },
  { actual: 6250, predicted: 5980 },
];

export const residualData = [
  { predicted: 1150, residual: 50 },
  { predicted: 2420, residual: -105 },
  { predicted: 910, residual: 40 },
  { predicted: 45, residual: -45 },
  { predicted: 4320, residual: 276 },
  { predicted: 2150, residual: -170 },
  { predicted: 760, residual: 65 },
  { predicted: 14890, residual: 344 },
  { predicted: 2310, residual: 120 },
  { predicted: 19100, residual: -870 },
  { predicted: 5980, residual: 270 },
];

export const histogramData = {
  Recency: [
    { range: "0-30", count: 845, isHighValue: 420 },
    { range: "31-60", count: 512, isHighValue: 190 },
    { range: "61-90", count: 320, isHighValue: 92 },
    { range: "91-120", count: 210, isHighValue: 45 },
    { range: "121-180", count: 180, isHighValue: 20 },
    { range: "181-270", count: 150, isHighValue: 10 },
    { range: "271-365", count: 432, isHighValue: 5 },
  ],
  Frequency: [
    { range: "1-2", count: 1250, isHighValue: 50 },
    { range: "3-5", count: 780, isHighValue: 180 },
    { range: "6-10", count: 390, isHighValue: 210 },
    { range: "11-20", count: 150, isHighValue: 120 },
    { range: "21-50", count: 62, isHighValue: 55 },
    { range: "51+", count: 17, isHighValue: 17 },
  ],
  Monetary: [
    { range: "0-500", count: 1540, isHighValue: 0 },
    { range: "501-1000", count: 620, isHighValue: 45 },
    { range: "1001-1629", count: 382, isHighValue: 120 },
    { range: "1630-3000", count: 310, isHighValue: 310 },
    { range: "3001-5000", count: 125, isHighValue: 125 },
    { range: "5001+", count: 72, isHighValue: 72 },
  ],
};
