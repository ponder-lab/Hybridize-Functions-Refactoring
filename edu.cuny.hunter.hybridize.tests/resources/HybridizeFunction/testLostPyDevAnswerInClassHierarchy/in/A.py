import lost_base_answer_aborts


class M(lost_base_answer_aborts.Base):
    def call(self, x):
        return x


print(M().call(1))
