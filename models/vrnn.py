import torch
import torch.nn as nn
from ..utils.functions import ReverseGradient


class VRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, n_labels, p_dropout=0.1):
        super().__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(x_dim, h_dim),
            nn.ReLU()
        )
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )

        # prior
        self.phi_prior = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim),
            nn.ReLU()
        )
        self.prior_mean = nn.Linear(z_dim, z_dim)
        self.prior_log_std = nn.Linear(z_dim, z_dim)
        # encode

        self.phi_encode = nn.Sequential(
            nn.Linear(h_dim*2, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim),
            nn.ReLU()
        )
        self.encode_mean = nn.Linear(z_dim, z_dim)
        self.encode_log_std = nn.Linear(z_dim, z_dim)

        # decode
        self.phi_decode(h_dim*2, x_dim)
        self.decode_mean = nn.Sequential(
            nn.Linear(x_dim, x_dim),
            nn.Tanh()
        )
        nn.Linear(x_dim, x_dim)
        self.decode_log_std = nn.Linear(x_dim, x_dim)

        # use GRUCell for recurrence
        self.rnn = nn.GRUCell(h_dim*2, h_dim)
        # class classifier
        self.class_classifier = nn.Sequential(
            nn.Linear(z_dim, z_dim//2),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(z_dim//2, z_dim//2),
            nn.ReLU(),
            nn.Linear(z_dim//2, n_labels),
            nn.Softmax()
        )
        # domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(z_dim, z_dim//4),
            nn.ReLU(),
            nn.Linear(z_dim//4, 1),
            nn.Sigmoid()
        )
        self.rcLoss = nn.MSELoss()

        def forward(self, train_input, test_input):
            """
            Args:
                    train_input (shape: (input_size,batch_size_train,features))
                    test_input (shape: (input_size,batch_size_test,features))
            """
            kl_loss = 0
            rc_loss =0
            batch_size_test = test_input.size(1)
            total_input = torch.cat([train_input, test_input], dim=1)
            hidden_features = torch.zeros(
                (total_input.size(0), total_input.size(1), z_dim))
            h = torch.zeros(total_input.size(1), self.h_dim)
            for t in range(total_input.size(0)):
                phi_x_t = self.phi_x(total_input[t])
                # encode
                enc_t = self.phi_encode(torch.cat(phi_x_t, h), 1)
                enc_mean_t = self.encode_mean(enc_t)
                enc_log_var_t = self.encode_log_std(enc_t)
                # prior
                prior_t = self.phi_prior(h)
                prior_mean_t = self.prior_mean(prior_t)
                prior_log_var_t = self.prior_log_std(prior_t)

                # reparameterization
                z_t = self._reparameterize(enc_mean_t, enc_log_var_t)
                hidden_features[:, t, :] = z_t
                phi_z_t = self.phi_z(z_t)
                # decode
                dec_t = self.decode(z_t)
                dec_mean_t = self.decode_mean(dec_t)
                # dec_log_var_t = self.decode_log_std(dec_t)

                # recurrence
                h = self.rnn(torch.cat([phi_x_t, phi_z_t], dim=1), h)

                # KL divergence loss
                kl_loss += self._kld_gauss(enc_mean_t,
                                           enc_log_var_t, prior_mean_t,
                                           prior_log_var_t)

                # reconstruction loss
                rc_loss += self.rc_loss(dec_mean_t, total_input[t])

            # domain classification
            neg_grad_hidden_features = ReverseGradient(hidden_features)
            domain_predicted = self.domain_classifier(
                neg_grad_hidden_features)
            #
            labels_predicted = self.class_classifier(
                hidden_features[:, batch_size_test:, :])
            return kl_loss, rc_loss, domain_predicted, labels_predicted

        def _reparameterize(self, mean, log_var):
            epsilon = torch.randn_like(mean)
            return mean+epsilon*(log_var/2).exp()

        def _kld_gauss(self, mean_1, log_var_1, mean_2, log_var_2):
            """Using std to compute KLD"""
            kld_element = (log_var_2 - log_var_1 +
                           (log_var_1.exp() + (mean_1 - mean_2).pow(2)) /
                           log_var_2.exp() - 1)
            return 0.5 * torch.sum(kld_element)
