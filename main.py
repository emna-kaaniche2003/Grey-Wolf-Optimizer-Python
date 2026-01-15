import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================
# CLASSE: GREY WOLF OPTIMIZER
# ==========================================
class GWO:
    def __init__(self, obj_func, lower_bound, upper_bound, dim, wolf_count, max_iter):
        """
        Initialise l'optimiseur GWO.
        
        :param obj_func: La fonction à minimiser (Fitness Function)
        :param lower_bound: Limite inférieure de l'espace de recherche (ex: -10)
        :param upper_bound: Limite supérieure de l'espace de recherche (ex: 10)
        :param dim: Nombre de dimensions (ex: 2 pour x,y ou 30 pour complexe)
        :param wolf_count: Nombre de loups dans la meute
        :param max_iter: Nombre maximum d'itérations
        """
        self.obj_func = obj_func
        self.lb = lower_bound
        self.ub = upper_bound
        self.dim = dim
        self.pop_size = wolf_count
        self.max_iter = max_iter
        
        # Initialisation des positions des loups (Aléatoire entre lb et ub)
        self.positions = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        
        # Initialisation des leaders (Alpha, Beta, Delta)
        # On met l'infini au début car on cherche à minimiser
        self.alpha_pos = np.zeros(self.dim)
        self.alpha_score = float("inf")
        
        self.beta_pos = np.zeros(self.dim)
        self.beta_score = float("inf")
        
        self.delta_pos = np.zeros(self.dim)
        self.delta_score = float("inf")
        
        # Historique pour le graphique de convergence
        self.convergence_curve = []

    def optimize(self):
        """
        Exécute la boucle principale de l'algorithme.
        """
        print(f"🚀 Démarrage de GWO sur {self.dim} dimensions pour {self.max_iter} itérations...")
        
        for t in range(self.max_iter):
            
            # --- 1. Évaluation de la Fitness ---
            for i in range(self.pop_size):
                
                # Vérifier que le loup reste dans les limites (Boundary Check)
                self.positions[i, :] = np.clip(self.positions[i, :], self.lb, self.ub)
                
                # Calcul du score
                fitness = self.obj_func(self.positions[i, :])
                
                # Mise à jour des leaders (Hiérarchie sociale)
                if fitness < self.alpha_score:
                    self.delta_score = self.beta_score  # Le Beta devient Delta
                    self.delta_pos = self.beta_pos.copy()
                    self.beta_score = self.alpha_score  # L'Alpha devient Beta
                    self.beta_pos = self.alpha_pos.copy()
                    self.alpha_score = fitness          # Nouveau Alpha
                    self.alpha_pos = self.positions[i, :].copy()
                    
                elif fitness > self.alpha_score and fitness < self.beta_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    self.beta_score = fitness
                    self.beta_pos = self.positions[i, :].copy()
                    
                elif fitness > self.alpha_score and fitness > self.beta_score and fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.positions[i, :].copy()
            
            # --- 2. Mise à jour du paramètre 'a' ---
            # 'a' diminue linéairement de 2 à 0
            a = 2 - t * (2 / self.max_iter)
            
            # --- 3. Mise à jour de la position des loups ---
            for i in range(self.pop_size):
                for j in range(self.dim):
                    
                    # --- FORMULES MATHÉMATIQUES (Slide 17-19) ---
                    
                    # Influence de l'Alpha
                    r1, r2 = np.random.random(), np.random.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[j] - self.positions[i, j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha
                    
                    # Influence du Beta
                    r1, r2 = np.random.random(), np.random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta_pos[j] - self.positions[i, j])
                    X2 = self.beta_pos[j] - A2 * D_beta
                    
                    # Influence du Delta
                    r1, r2 = np.random.random(), np.random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta_pos[j] - self.positions[i, j])
                    X3 = self.delta_pos[j] - A3 * D_delta
                    
                    # Position finale (Moyenne)
                    self.positions[i, j] = (X1 + X2 + X3) / 3
            
            # Enregistrement pour le graphique
            self.convergence_curve.append(self.alpha_score)
            
            # Affichage console tous les 10 tours
            if t % 10 == 0:
                print(f"Iteration {t}: Best Fitness = {self.alpha_score:.6f}")

        print("\n✅ Optimisation terminée.")
        print(f"Meilleure position trouvée : {self.alpha_pos}")
        print(f"Meilleur score (Fitness) : {self.alpha_score}")
        
        return self.convergence_curve

# ==========================================
# FONCTION OBJECTIF (Le problème à résoudre)
# ==========================================
def sphere_function(x):
    """
    Fonction Sphère : f(x) = sum(x^2)
    Le minimum global est 0 à la position [0, 0, ..., 0]
    """
    return np.sum(x**2)

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # --- Paramètres de la simulation ---
    search_agents = 10      # Nombre de loups
    max_iterations = 50     # Nombre de cycles
    dimensions = 2          # 2D pour pouvoir visualiser (x, y)
    lower_bound = -10
    upper_bound = 10
    
    # --- Création et Lancement de l'Optimiseur ---
    optimizer = GWO(sphere_function, lower_bound, upper_bound, dimensions, search_agents, max_iterations)
    convergence = optimizer.optimize()
    
    # --- Visualisation des Résultats ---
    plt.figure(figsize=(12, 5))

    # Graphique 1 : Courbe de Convergence
    plt.subplot(1, 2, 1)
    plt.plot(convergence, color='red', linewidth=2)
    plt.title('Convergence Curve (Evolution du Fitness)')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness (Meilleur Score)')
    plt.grid(True)

    # Graphique 2 : Visualisation de la position finale
    # Note: On visualise seulement le point trouvé vs le centre
    plt.subplot(1, 2, 2)
    plt.title('Espace de Recherche 2D')
    plt.xlim(lower_bound, upper_bound)
    plt.ylim(lower_bound, upper_bound)
    plt.scatter(0, 0, c='green', s=200, marker='*', label='Proie (Optimum 0,0)')
    plt.scatter(optimizer.alpha_pos[0], optimizer.alpha_pos[1], c='red', s=100, label='Alpha (Trouvé)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()