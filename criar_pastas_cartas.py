import os

def criar_pastas_cartas():
    """
    Cria pastas organizadas para cada carta do baralho
    Estrutura: cards_templates/[carta]/[carta]_[numero].png
    """
    
    # Naipes
    naipes = ['s', 'h', 'd', 'c']  # Espadas, Copas, Diamantes, Paus
    
    # Cartas (A, K, Q, J, T, 9, 8, 7, 6, 5, 4, 3, 2)
    cartas = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    
    pasta_principal = "cards_templates"
    
    print("=== CRIANDO PASTAS ORGANIZADAS ===")
    print(f"Criando estrutura em: {pasta_principal}")
    
    # Criar pasta principal se não existir
    if not os.path.exists(pasta_principal):
        os.makedirs(pasta_principal)
        print(f"✓ Pasta principal criada: {pasta_principal}")
    
    # Criar pasta para cada carta
    cartas_criadas = []
    
    for carta in cartas:
        for naipe in naipes:
            nome_carta = f"{carta}{naipe}"
            pasta_carta = os.path.join(pasta_principal, nome_carta)
            
            if not os.path.exists(pasta_carta):
                os.makedirs(pasta_carta)
                cartas_criadas.append(nome_carta)
                print(f"✓ Pasta criada: {nome_carta}")
    
    print(f"\n=== RESUMO ===")
    print(f"Total de cartas: {len(cartas) * len(naipes)} (52 cartas)")
    print(f"Pastas criadas: {len(cartas_criadas)}")
    
    if cartas_criadas:
        print("\nCartas novas:")
        for carta in cartas_criadas[:10]:  # Mostrar primeiras 10
            print(f"  - {carta}")
        if len(cartas_criadas) > 10:
            print(f"  ... e mais {len(cartas_criadas) - 10} cartas")
    
    print(f"\nPronto! Agora você pode colocar as imagens de cada carta na pasta correspondente.")
    print(f"Exemplo: cards_templates/As/As_1.png, cards_templates/As/As_2.png")

if __name__ == "__main__":
    criar_pastas_cartas()
    input("\nPressione ENTER para sair...")
