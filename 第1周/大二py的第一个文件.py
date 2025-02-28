import numpy as np
import matplotlib.pyplot as plt

# �����˹�ֲ��Ĳ���
mu, sigma = 0, 0.1  # ��ֵ�ͱ�׼��

# ��������
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
y = (1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

# ���Ƹ�˹�ֲ�����
plt.plot(x, y, label=f'Gaussian ($\mu={mu}$, $\sigma={sigma}$)')
plt.title('Gaussian Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()