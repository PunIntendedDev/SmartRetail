import axios from 'axios';

// Configure Axios Client targeting a future Python backend (e.g., FastAPI / Flask)
const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api',
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000, // 10 seconds timeout
});

/**
 * Predicts customer segment classification and forecasted spend (LTV)
 * @param {Object} customerData - Object containing RFM and spend percentage traits
 * @returns {Promise} Axios response promise
 */
export const predictCustomer = async (customerData) => {
  // Placeholder structure - ready to be swapped with backend integration:
  // return apiClient.post('/predict', customerData);
  
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        data: {
          prediction: customerData.monetary >= 1629.40 ? 'High Value' : 'Standard',
          confidence: customerData.monetary >= 1629.40 ? 98.7 : 94.2,
          futureSpend: (customerData.monetary * 0.45).toFixed(2),
          riskLevel: customerData.monetary >= 1629.40 ? 'Low' : 'Medium',
          recommendation: customerData.monetary >= 1629.40
            ? 'Assign to premium loyalty benefits and priority service representative.'
            : 'Target with promo vouchers to stimulate repeat basket orders.',
        }
      });
    }, 500); // Simulate network latency
  });
};

/**
 * Fetches trained models accuracy, regression R2, and evaluation plots datasets
 * @returns {Promise} Axios response promise
 */
export const getAnalytics = async () => {
  // return apiClient.get('/analytics');
  
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        data: {
          status: 'success',
          classification: { accuracy: 1.0, precision: 1.0, recall: 1.0, f1: 1.0, auc: 1.0 },
          regression: { mse: 13307866.38, rmse: 3647.99, r2: 0.6593 },
        }
      });
    }, 300);
  });
};

/**
 * Retrieves paginated dataset records for explorer views
 * @param {number} page - Page number
 * @param {number} limit - Items per page
 * @returns {Promise} Axios response promise
 */
export const getDataset = async (page = 1, limit = 10) => {
  // return apiClient.get('/dataset', { params: { page, limit } });
  
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        data: {
          status: 'success',
          totalRecords: 3312,
          page,
          limit,
        }
      });
    }, 400);
  });
};

export default apiClient;
