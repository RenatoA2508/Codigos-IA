import tkinter as tk
from tkinter import ttk

from nim import Nim, train


TRAINING_GAMES = 10000


class NimApp:
    """
    Basic graphical interface for playing Nim against a trained AI.
    """

    def __init__(self, root, ai):
        self.root = root
        self.ai = ai
        self.game = Nim()
        self.human_player = 0
        self.last_ai_move = None

        self.root.title("Nim Q-Learning")
        self.root.resizable(False, False)

        self.status_var = tk.StringVar()
        self.pile_frame = ttk.Frame(root, padding=12)
        self.pile_frame.grid(row=0, column=0, sticky="nsew")

        self.status = ttk.Label(root, textvariable=self.status_var, padding=(12, 0, 12, 8))
        self.status.grid(row=1, column=0, sticky="w")

        controls = ttk.Frame(root, padding=(12, 0, 12, 12))
        controls.grid(row=2, column=0, sticky="ew")

        self.restart_button = ttk.Button(controls, text="Nuevo juego", command=self.restart)
        self.restart_button.grid(row=0, column=0, sticky="w")

        q_frame = ttk.LabelFrame(root, text="Valores Q aprendidos", padding=12)
        q_frame.grid(row=0, column=1, rowspan=3, sticky="nsew", padx=(0, 12), pady=12)

        columns = ("state", "action", "value")
        self.q_table = ttk.Treeview(q_frame, columns=columns, show="headings", height=18)
        self.q_table.heading("state", text="Estado")
        self.q_table.heading("action", text="Accion")
        self.q_table.heading("value", text="Q")
        self.q_table.column("state", width=130, anchor="w")
        self.q_table.column("action", width=80, anchor="center")
        self.q_table.column("value", width=70, anchor="e")
        self.q_table.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(q_frame, orient="vertical", command=self.q_table.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.q_table.configure(yscrollcommand=scrollbar.set)

        self.populate_q_table()
        self.render_game()

    def populate_q_table(self):
        """
        Show learned Q-values after training, sorted from highest to lowest.
        """
        ranked_q_values = sorted(
            self.ai.q.items(),
            key=lambda item: item[1],
            reverse=True
        )
        for (state, action), value in ranked_q_values:
            self.q_table.insert("", "end", values=(state, action, f"{value:.3f}"))

    def restart(self):
        self.game = Nim()
        self.last_ai_move = None
        self.render_game()

    def render_game(self):
        for widget in self.pile_frame.winfo_children():
            widget.destroy()

        ttk.Label(self.pile_frame, text="Pilas").grid(row=0, column=0, sticky="w")

        for row, pile in enumerate(self.game.piles, start=1):
            ttk.Label(self.pile_frame, text=f"Pila {row - 1}: {pile}").grid(
                row=row,
                column=0,
                sticky="w",
                padx=(0, 10),
                pady=4
            )

            for count in range(1, pile + 1):
                button = ttk.Button(
                    self.pile_frame,
                    text=str(count),
                    width=3,
                    command=lambda pile_index=row - 1, amount=count: self.human_move(pile_index, amount)
                )
                if self.game.player != self.human_player or self.game.winner is not None:
                    button.state(["disabled"])
                button.grid(row=row, column=count, padx=2, pady=4)

        if self.game.winner is None:
            if self.game.player == self.human_player:
                if self.last_ai_move is None:
                    self.status_var.set("Tu turno: elige cuantos objetos tomar de una pila.")
                else:
                    pile, count = self.last_ai_move
                    self.status_var.set(f"La IA tomo {count} de la pila {pile}. Tu turno.")
            else:
                self.status_var.set("Turno de la IA...")
        else:
            winner = "Humano" if self.game.winner == self.human_player else "IA"
            self.status_var.set(f"Juego terminado. Ganador: {winner}.")

    def human_move(self, pile, count):
        if self.game.winner is not None or self.game.player != self.human_player:
            return

        self.last_ai_move = None
        self.game.move((pile, count))
        if self.game.winner is not None:
            self.render_game()
            return

        self.render_game()
        self.root.after(500, self.ai_move)

    def ai_move(self):
        if self.game.winner is not None or self.game.player == self.human_player:
            return

        action = self.ai.choose_action(self.game.piles, epsilon=False)
        if action is None:
            return

        pile, count = action
        self.game.move(action)
        self.last_ai_move = action
        self.render_game()


def main():
    ai = train(TRAINING_GAMES)

    root = tk.Tk()
    NimApp(root, ai)
    root.mainloop()


if __name__ == "__main__":
    main()
